import torch
import torch.nn.functional as F
import torchvision.models as models
import argparse
import os

import json

from utils import *
from cam.layercam import *
from torchvision.models import VGG16_Weights

def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of LayerCAM')

    #parser.add_argument("--img_path", type=str, default='images/bulbul.JPEG', help='Path of test image') # bird
    #parser.add_argument("--img_path", type=str, default='images/bullet_train.JPEG', help='Path of test image') # train
    #parser.add_argument("--img_path", type=str, default='images/fox.JPEG', help='Path of test image')
    parser.add_argument("--img_path", type=str, default='images/dogs.JPEG', help='Path of test image')
    #parser.add_argument("--img_path", type=str, default='images/car.png', help='Path of test image')
    #parser.add_argument("--img_path", type=str, default='images/mountain2.png', help='Path of test image')

    parser.add_argument("--layer_id", type=list, default=[4,9,16,23,30], help='The cam generation layer') 
    return parser.parse_args()

import torch

# TODO text/fix
def run_layercam_for_label(input_, vgg_layercam, imagenet_class_idx, label_name):
    class_idx = None
    for k, v in imagenet_class_idx.items():
        if v[1] == label_name:
            class_idx = int(k)
            break

    if class_idx is None:
        raise ValueError(f"Label {label_name} not found in the ImageNet index.")

    norm_cam = vgg_layercam(input_, class_idx=class_idx)

    return norm_cam


if __name__ == '__main__':
    args = get_arguments()

    # Load the imagenet class index
    json_path = os.path.join("utils", "imagenet_class_index.json")
    with open(json_path, 'r') as f:
        imagenet_class_idx = json.load(f)


    input_image = load_image(args.img_path)
    input_ = apply_transforms(input_image)
    if torch.cuda.is_available():
        input_ = input_.cuda()

  
    # Extract the base name of the image without extension
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]

    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
    maps = []
    for i, layer_idx in enumerate(args.layer_id, 1):
        layer_name = 'features_' + str(layer_idx)
        vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name=layer_name, input_size=(224, 224))
        vgg_layercam = LayerCAM(vgg_model_dict)
        predicted_class = vgg(input_).max(1)[-1].item()

        #label_name = "goldfish"  
        #cam_map = run_layercam_for_label(input_, vgg_layercam, imagenet_class_idx, label_name)

        predicted_label = imagenet_class_idx[str(predicted_class)][1]
        print("Predicted label name:", predicted_label, " Index: ", predicted_class)

        layercam_map = vgg_layercam(input_)
        maps.append(layercam_map)

        # Use the base_name in the output filenames
        basic_visualize_separate(
            input_.cpu().detach(),
            layercam_map.type(torch.FloatTensor).cpu(),
            save_path=f'./vis/{base_name}_stage_{i}'
        )

    basic_visualize_sequence(input_.cpu().detach(), maps, save_path=f'./vis/{base_name}_all_stages.png')

    # For max combination progression
    basic_visualize_combination_progression(
        input_.cpu().detach(),
        maps,
        operation='max',
        save_path=f'./vis/{base_name}_combination_progression_max.png'
    )

    basic_visualize_combination_progression_reverse(
        input_.cpu().detach(),
        maps,
        operation='max',
        save_path=f'./vis/{base_name}_combination_progression_max_reverse.png'
    )

    # For add combination progression
    basic_visualize_combination_progression(
        input_.cpu().detach(),
        maps,
        operation='add',
        save_path=f'./vis/{base_name}_combination_progression_add.png'
    )
    basic_visualize_combination_progression_reverse(
        input_.cpu().detach(),
        maps,
        operation='add',
        save_path=f'./vis/{base_name}_combination_progression_add_reverse.png'
    )
    # For mul combination progression
    basic_visualize_combination_progression(
        input_.cpu().detach(),
        maps,
        operation='mul',
        save_path=f'./vis/{base_name}_combination_progression_mul.png'
    )
    basic_visualize_combination_progression_reverse(
        input_.cpu().detach(),
        maps,
        operation='mul',
        save_path=f'./vis/{base_name}_combination_progression_mul_reverse.png'
    )
    exit()

    # Combine maps using different operations
    combined_max_map = combine_maps(maps, operation='max')
    combined_add_map = combine_maps(maps, operation='add')
    combined_mul_map = combine_maps(maps, operation='mul')

    # Visualize the combined maps, also including the base name
    basic_visualize_separate(
        input_.cpu().detach(), 
        combined_max_map.cpu(),
        save_path=f'./vis/{base_name}_combined_max'
    )

    basic_visualize_separate(
        input_.cpu().detach(), 
        combined_add_map.cpu(), 
        save_path=f'./vis/{base_name}_combined_add'
    )

    basic_visualize_separate(
        input_.cpu().detach(), 
        combined_mul_map.cpu(), 
        save_path=f'./vis/{base_name}_combined_mul'
    )
