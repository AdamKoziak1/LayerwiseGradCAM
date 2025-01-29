import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, \
    find_squeezenet_layer, find_layer, find_googlenet_layer, find_mobilenet_layer, find_shufflenet_layer

from cam.basecam import BaseCAM



class CAMFormer(BaseCAM):
    """
    Example of a CAM-like Transformer that:
      • Gathers multi-layer activations and gradients from a backbone (e.g. VGG).
      • Encodes each layer’s feature map into a shared latent dimension.
      • Appends a class label token to the sequence.
      • Applies a Transformer to fuse information.
      • Decodes back into masks or gating coefficients for each layer.
    """

    def __init__(
        self,
        model_dict,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=1000  # adapt based on your dataset
    ):
        
        """
        :param model_dict: same as in BaseCAM, with keys for 'type', 'arch', 'layer_names'
        :param hidden_dim: dimension of the Transformer embeddings
        :param num_heads: number of attention heads in multi-head attention
        :param num_layers: number of Transformer encoder/decoder layers
        :param num_classes: how many classes you expect (for the class token embedding)
        """
        super().__init__(model_dict)
        device = next(self.model_arch.parameters()).device
        
        self.hidden_dim = hidden_dim
        # We'll collect shapes in a dict: {layer_name: (out_channels, out_h, out_w)}
        self.layer_shapes = {}

        self.layer_names = model_dict['layer_names']

        hook_handles = []
        # 1) Temporarily install a forward hook to find shape
        def shape_forward_hook(name):
            def hook(module, input, output):
                # output shape e.g. [B, C, H, W]
                b, c, h, w = output.shape
                self.layer_shapes[name] = (c, h, w)
            return hook
        
        # 1) Temporarily install a forward hook to find shape for each layer
        for layer_name in self.layer_names:
            layer = find_vgg_layer(self.model_arch, layer_name)
            handle = layer.register_forward_hook(shape_forward_hook(layer_name))
            hook_handles.append(handle)

        # 2) Do a dummy forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        _ = self.model_arch(dummy_input)

        for key in self.layer_shapes:
            print(key, self.layer_shapes[key])
       # 3) Remove the shape-only hooks
        for handle in hook_handles:
            handle.remove()


        # Now self.layer_shapes should have the channel count for each layer

        # 3) Build the actual encoders/decoders using the recorded out_channels
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for layer_name in self.layer_names:
            print(layer_name)
            c, _, _ = self.layer_shapes[layer_name]
            # c is the channel count
            encoder = nn.Conv2d(c, hidden_dim, kernel_size=1).to(device)
            self.encoders.append(encoder)
            decoder = nn.Conv2d(hidden_dim, c, kernel_size=1).to(device)
            self.decoders.append(decoder)
        print('enc', len(self.encoders), 'dec', len(self.decoders))
        # 4) Remove or override the shape-only hooks if needed, or let them stay
        #    Then re-register your actual Grad-CAM forward/backward hooks
        #    But in practice, you might do everything in the same hook if you want.

        # ... also define your Transformer, class embeddings, etc.


        # Class token embedding
        # If you have multiple classes, you can embed the class index directly
        # or keep a single token if you always look at a single “target” class
        self.class_emb = nn.Embedding(num_classes, hidden_dim).to(device)

        # Transformer: you can use the standard PyTorch nn.TransformerEncoder or a custom approach
        # Here, we define an encoder-only Transformer for self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        :param input: a 4D input tensor (B x C x H x W)
        :param class_idx: which class to target, if not None
        :param retain_graph: whether to retain the graph for subsequent backward passes
        """
        print('----FORWARD----')

        # 1) Forward pass through the backbone (VGG) to populate self.activations and self.gradients
        logit = self.model_arch(input)  # shape [B, #classes]
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, predicted_class].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx]*input.size(0))
            score = logit[:, class_idx].squeeze()

        print('logit', logit.shape)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            score = score.cuda()

        # 2) Backward pass to get the gradients for the chosen class
        one_hot_output = torch.zeros_like(logit)
        for b_idx in range(input.size(0)):
            one_hot_output[b_idx, predicted_class[b_idx]] = 1
        self.model_arch.zero_grad()
        logit.backward(gradient=one_hot_output, retain_graph=retain_graph)

        # Now self.activations and self.gradients hold feature maps (and grads) from each target layer
        # Remember that we appended them in forward order, so you may need to track which is which
        # or rely on the order in which you registered hooks.

        # We’ll do a simple example: apply our Transformer to the activations alone
        # You could also incorporate gradients or grad × activation as well.

        # Reverse the lists if needed, or track them carefully
        # For simplicity, assume self.activations are in the same order as self.layer_names
        # If not, reorder them accordingly
        # We’ll encode each layer’s activation, flatten it, then gather them into a single sequence
        layer_tokens = []
        batch_size = input.size(0)

        for i, (activation, encoder) in enumerate(zip(self.activations, self.encoders)):
            print('enumerate(zip(self.activations, self.encoders))', i)
            # activation shape: [B, C, H, W]
            # 1) Project channels to hidden_dim
            encoded = encoder(activation)  # [B, hidden_dim, H, W]
            print(encoded.shape)
            # 2) Flatten spatially so each position is a “token”
            # shape => [B, hidden_dim, H*W]
            encoded = encoded.flatten(2, 3)
            print(encoded.shape)
            # Then transpose so tokens dimension is second: [B, H*W, hidden_dim]
            encoded = encoded.transpose(1, 2)
            print(encoded.shape)
            # Store
            layer_tokens.append(encoded)
            

        # Concatenate all layer tokens into one sequence
        # shape => [B, total_tokens, hidden_dim]
        all_tokens = torch.cat(layer_tokens, dim=1)
        print('all tokens', all_tokens.shape)

        # 3) Append class token
        # If you have multiple classes in the same batch, gather class embeddings accordingly
        class_embs = self.class_emb(predicted_class)  # [B, hidden_dim]
        print('pred class', predicted_class.shape)
        # Expand to [B, 1, hidden_dim]
        class_embs = class_embs.unsqueeze(1)
        print('class_embs', class_embs.shape)
        # Now cat with all_tokens => shape [B, total_tokens + 1, hidden_dim]
        all_tokens = torch.cat([class_embs, all_tokens], dim=1)
        print('all tokens and class', all_tokens.shape)

        # 4) Transformer self-attention
        # The standard nn.TransformerEncoder expects [B, seq_len, d_model] if batch_first=True
        transformed = self.transformer_encoder(all_tokens)  
        print('transformed', transformed.shape)
        # shape => [B, total_tokens + 1, hidden_dim]

        # The class token is now at transformed[:, 0, :] if you want to do classification
        # or you can ignore it for segmentation masks. Let’s skip using it further here.

        # 5) Split the transformed sequence back per layer to decode
        # The first token is the class token, so skip index 0
        offset = 1
        reconstructed_masks = []

        for i, decoder in enumerate(self.decoders):
            print('-----',i,'-----')
            # Each layer had shape [B, H_i * W_i, hidden_dim]
            original_num_tokens = layer_tokens[i].size(1)
            print('orig num tokens', original_num_tokens)
            # Extract the relevant slice from transformed (skip the first class token)
            layer_slice = transformed[:, offset:(offset + original_num_tokens), :]
            offset += original_num_tokens

            # Reshape back to [B, hidden_dim, H_i, W_i]
            bsz, ntok, dim = layer_slice.shape
            # from [B, ntok, hidden_dim] to [B, hidden_dim, ntok]
            layer_slice = layer_slice.transpose(1, 2)
            # Suppose we know H_i, W_i from the activation shape
            # You can track them or infer them from self.activations[i].shape
            # Let’s do it quickly:
            _, c_i, h_i, w_i = self.activations[i].shape
            layer_slice = layer_slice.view(bsz, dim, h_i, w_i)

            # 6) Decode back to the original channel dimension
            # shape => [B, out_channels (like c_i), h_i, w_i]
            decoded = decoder(layer_slice)
            print('decoded', decoded.shape)

            # You can interpret 'decoded' as a gating mask, or channels for further usage
            # For a gating mask, you might reduce the channel dimension to 1
            # For now, let’s assume the decoder out_channels = c_i
            # So we can do elementwise multiplication with the original activation
            # or pass it into a segmentation head, etc.
            reconstructed_masks.append(decoded)
        print('len(reconstructed_masks)', len(reconstructed_masks))
        # You now have a list of decoded outputs, each shaped like [B, c_i, H_i, W_i].
        # For example, to get a gating mask, you might do:
        # gating_mask = torch.sigmoid(reconstructed_masks[i])  # shape [B, c_i, H_i, W_i]
        # and multiply with the original activation or do a direct segmentation loss.

        # Clear activations if you want to avoid memory growth on repeated calls
        self.activations = []
        self.gradients = []

        # Return your final masks or gating outputs. You could also return the classification logit.
        return reconstructed_masks

    def __call__(self, input, class_idx=None, retain_graph=False):
        # Reset stored gradients and activations
        self.gradients = []
        self.activations = []
        return self.forward(input, class_idx, retain_graph)
