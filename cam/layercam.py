import torch
import torch.nn.functional as F
from cam.basecam import *


class LayerCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM
    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        
        # predication on raw input
        logit = self.model_arch(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        #logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class = predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()
        
        one_hot_output = torch.FloatTensor(1, logit.size()[-1]).zero_()
        one_hot_output[0][predicted_class] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        self.model_arch.zero_grad()
        # Backward pass with specified target
        logit.backward(gradient=one_hot_output, retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        activations.reverse()
        # now we have gradients and activations across all layers and channels

        # TODO attention over those 

        #LayerCAM:
        norm_cams = []
        with torch.no_grad():
            for i in range(len(activations)):
                activation_maps = activations[i] * F.relu(gradients[i])
                cam = torch.sum(activation_maps, dim=1).unsqueeze(0)    
                cam = F.interpolate(cam, size=(h, w), mode='bicubic', align_corners=False)      
                cam_min, cam_max = cam.min(), cam.max()
                norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
                norm_cams.append(norm_cam)
        return norm_cams

    def __call__(self, input, class_idx=None, retain_graph=False):
        self.gradients = []
        self.activations = []
        return self.forward(input, class_idx, retain_graph)