import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.models import VGG16_Weights

def find_vgg_layer(model, layer_name):
    #This function should return the nn.Module corresponding to layer_name, such as "features.29" in vgg16, etc.
    return dict(model.named_modules())[layer_name]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CAMFormerModule(nn.Module):
    """
    A self-contained module combining:
      1) A backbone model (e.g., VGG16).
      2) Hooks to capture activations and gradients for Grad-CAM.
      3) A Transformer-based encoder-decoder mechanism to fuse multi-layer features.
      4) An optional call to do a Grad-CAM style backward internally.
    """

    def __init__(
        self,
        layer_names,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        num_classes=1000,
    ):
        """
        :param backbone: a pretrained model like vgg16(pretrained=True).
        :param layer_names: list of layer names (strings) for hooking.
        :param hidden_dim: dimension of Transformer embeddings.
        :param num_heads: number of attention heads in multi-head attention.
        :param num_layers: number of TransformerEncoder layers.
        :param num_classes: for class embedding if Grad-CAM style usage is desired.
        """
        super().__init__()

        backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()

        self.model_arch = backbone.eval()  # Keep in eval mode for hooking, or remove if training end-to-end
        self.layer_names = layer_names
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate=dropout_rate

        # Move the backbone to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_arch.to(self.device)

        # Containers for storing per-layer activations and gradients (Grad-CAM style)
        self.activations = []
        self.gradients = []

        # Step 1: Temporarily register forward hooks to find each layer's shape
        self.layer_shapes = {}
        hook_handles = []

        def shape_forward_hook(layer_id):
            def hook(module, inp, out):
                # out shape => [B, C, H, W]
                _, c, h, w = out.shape
                self.layer_shapes[layer_id] = (c, h, w)
            return hook

        # Attach shape-only hooks
        for layer_n in self.layer_names:
            layer = find_vgg_layer(self.model_arch, layer_n)
            h = layer.register_forward_hook(shape_forward_hook(layer_n))
            hook_handles.append(h)

        # Do a dummy forward pass
        dummy_x = torch.randn(1, 3, 224, 224, device=self.device)
        _ = self.model_arch(dummy_x)

        # Remove the shape hooks
        for h in hook_handles:
            h.remove()

        # Step 2: Build per-layer encoders/decoders
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for layer_n in self.layer_names:
            c, _, _ = self.layer_shapes[layer_n]
            enc = nn.Conv2d(c, hidden_dim, kernel_size=1, dtype=torch.bfloat16)
            dec = nn.Conv2d(hidden_dim, c, kernel_size=1, dtype=torch.bfloat16)
            self.encoders.append(enc)
            self.decoders.append(dec)

        # Step 3: Build a class embedding for Grad-CAM usage (if needed)
        self.class_emb = nn.Embedding(num_classes, hidden_dim, dtype=torch.bfloat16)

        # Step 4: Build a TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=self.dropout_rate,
            batch_first=True, 
            dtype=torch.bfloat16
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward_hook_fn(module, inp, out):
            self.activations.append(out.clone().detach().requires_grad_(True))

        def backward_hook_fn(module, grad_in, grad_out):
            self.gradients.append(grad_out[0].clone().detach().requires_grad_(True))


        self.target_layers = []
        for layer_n in self.layer_names:
            lyr = find_vgg_layer(self.model_arch, layer_n)
            self.target_layers.append(lyr)
            lyr.register_forward_hook(forward_hook_fn)
            lyr.register_backward_hook(backward_hook_fn)

        # Move entire submodules to the same device
        self.encoders.to(self.device)
        self.decoders.to(self.device)
        self.class_emb.to(self.device)
        self.transformer_encoder.to(self.device)


    def forward(self, x, class_idx=None, do_gradcam=True, retain_graph=False, eval=False):
        """
        :param x: input image batch, shape [B, 3, H, W].
        :param class_idx: optional integer specifying which class to "Grad-CAM" for.
        :param do_gradcam: if True, we do an internal backward pass to gather gradients.
        :param retain_graph: whether to retain the graph after backward (Grad-CAM usage).
        :return: a list of reconstructed layer outputs (like gating masks), one per target layer.
        """
        x = x.to(self.device)       
        self._clear_hooks_storage()  # Clear old activations/gradients       
        if do_gradcam:
            with torch.enable_grad():
                logits = self.model_arch(x)  # shape [B, #classes]       
                # Identify the target class for Grad-CAM
                if class_idx is None:
                    # For each sample, pick the top predicted class
                    predicted_class = logits.argmax(dim=1)  # [B]
                else:
                    # Single class for the whole batch
                    predicted_class = torch.LongTensor([class_idx]*x.size(0)).to(self.device)
        
                # Build one-hot
                oh = torch.zeros_like(logits)
                for b_idx in range(x.size(0)):
                    oh[b_idx, predicted_class[b_idx]] = 1

                # Zero grads and do backward
                self.model_arch.zero_grad()
                logits.backward(gradient=oh, retain_graph=retain_graph)

                #reset grad so VGG isn't trained
                self.model_arch.zero_grad()

                self.gradients.reverse()


        # 2) Encode each layer’s activations into hidden_dim tokens
        layer_tokens = []
        for act, enc in zip(self.activations, self.encoders):
            act = act.to(self.device)
            h_enc = enc(act)  # [B, hidden_dim, H, W]
            # Flatten
            bsz, d, hh, ww = h_enc.shape
            h_enc = h_enc.view(bsz, d, hh*ww)  # [B, d, hh*ww]
            h_enc = h_enc.transpose(1, 2)      # [B, hh*ww, d]
            layer_tokens.append(h_enc)

        # Concatenate tokens from all layers => [B, sumTokens, d]
        all_tokens = torch.cat(layer_tokens, dim=1)

        # 3) Prepend class token
        # For a single class per batch item, embed that class
        if do_gradcam:
            # If do_gradcam=True, we used predicted_class or class_idx
            class_embs = self.class_emb(predicted_class)  # [B, d]
        else:
            # If user turns off do_gradcam, default to class 0 or some placeholder
            class_embs = self.class_emb(torch.zeros(x.size(0), dtype=torch.long, device=self.device))
        class_embs = class_embs.unsqueeze(1)              # => [B, 1, d]
        all_tokens = torch.cat([class_embs, all_tokens], dim=1)  # => [B, sumTokens+1, d]

        # 4) Transformer
        transformed = self.transformer_encoder(all_tokens)  # => [B, sumTokens+1, d]

        # 5) Split back into layer slices and decode
        offset = 1
        reconstructed = []
        masks = []
        
        b_x, c_x, h_x, w_x = x.shape
        cam = torch.zeros((b_x,1,h_x,w_x), dtype=torch.float, device=self.device)

        for (act, dec) in zip(self.activations, self.decoders):
            bsz, c_i, h_i, w_i = act.shape
            n_tokens = h_i * w_i
            slice_ = transformed[:, offset:offset+n_tokens, :]
            offset += n_tokens

            # Reshape
            slice_ = slice_.transpose(1, 2)         # => [B, d, n_tokens]
            slice_ = slice_.view(bsz, self.hidden_dim, h_i, w_i)
            decoded = dec(slice_)                  # => [B, c_i, h_i, w_i]
            reconstructed.append(decoded)

            channel_sum = decoded.sum(axis=1).unsqueeze(1)
            masks.append(channel_sum)

            cam += F.interpolate(channel_sum, size=(224,224), mode='bicubic', align_corners=False) 
        cam_min, cam_max = cam.min(), cam.max()
        norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        norm_cam = norm_cam.squeeze(1)
        # cam: [B,1,224,224]

        output_dict = dict(act=self.activations, grad=self.gradients, masks=reconstructed, cam=norm_cam)
        return output_dict

    def _clear_hooks_storage(self):
        """
        Helper: clear out self.activations and self.gradients between forward passes.
        """
        self.activations = []
        self.gradients = []

    def count_params(self):
        backbone_params = count_parameters(self.model_arch)
        #backbone_params = 0
        enc_params = count_parameters(self.encoders)
        dec_params = count_parameters(self.decoders)
        emb_params = count_parameters(self.class_emb)
        trans_params = count_parameters(self.transformer_encoder)
        total = backbone_params + enc_params + dec_params + emb_params + trans_params
        print(f"Backbone params: {backbone_params}")
        print(f"Encoders params: {enc_params}")
        print(f"Decoders params: {dec_params}")
        print(f"Embedding params: {emb_params}")
        print(f"Transformer params: {trans_params}")
        print(f"Total params: {total}")
        return total

if __name__ == "__main__":
    layer_names = ["features.4", "features.9", "features.16", "features.23", "features.30"] # all max pool layers

    model = CAMFormerModule(
        layer_names=layer_names,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=1000
    )

    # Example usage
    images = torch.randn(5, 3, 224, 224).cuda()

    output_dict = model(images, class_idx=None, do_gradcam=True)  # returns a list of per-layer [B, c_i, H, W]
    activations = output_dict["act"]
    gradients = output_dict["grad"]
    recon = output_dict["masks"]
    cam = output_dict["cam"]

    
    print(len(recon))
    for i in range(len(recon)):
        print(activations[i].shape, gradients[i].shape, recon[i].shape)

    print(cam.shape)
