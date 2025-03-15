import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Directly return the classification output
        return self.model(x)
    
def create_gradcam(model, layers, DEVICE):
    model_wrapper = ModelWrapper(model)
    model_wrapper.to(DEVICE)
    model_wrapper.eval()
    # Target the bottleneck layer from the encoder as in the original code.
    if layers.startswith('densenet'):
        target_layer = [model_wrapper.model.encoder.features.norm5]
    elif layers.startswith('resnext') or layers.startswith('se_resnext'):
        target_layer = [model_wrapper.model.encoder.layer4[-1].bn2]
    elif layers.startswith('se_resnet'):
        target_layer = [model_wrapper.model.encoder.layer4[-1].bn2]
    elif layers == 'inceptionresnetv2':
        target_layer = [model_wrapper.model.encoder.mixed_7a]
    elif layers.startswith('mit_b'):
        target_layer = [model_wrapper.model.encoder.blocks[-1].norm1]
    elif layers.startswith('vgg'):
        target_layer = [model_wrapper.model.encoder.features[-1]]
    elif layers == 'inceptionv4':
        target_layer = [model_wrapper.model.encoder.features[-1]]
    elif layers.startswith('efficientnet'):
        target_layer = [model_wrapper.model.encoder._blocks[7][-1]]
    elif layers.startswith('resnet'):
        target_layer = [model_wrapper.model.encoder.layer4[-1]]
    else:
        target_layer = [model_wrapper.model.encoder.layer4[-1]]
    return GradCAM(model=model_wrapper, target_layers=target_layer)

def process_gradcam_batch(model, inputs, grad_cam, DEVICE, threshold, rounding_precision=5):
    """
    Run inference on one batch and compute Grad-CAM for classification.
    """
    inputs = inputs.to(DEVICE)
    with torch.no_grad():
        cls_pred = model(inputs)
        cls_pred = torch.sigmoid(cls_pred)
        cls_pred_bin = (cls_pred > threshold).float()
        cls_pred_rounded = round(cls_pred.item(), rounding_precision)

    # Remove the batch dimension and prepare the input image.
    inputs_squeezed = inputs.squeeze(0)
    inputs_np = np.transpose(inputs_squeezed.cpu().numpy(), (1, 2, 0))

    # Compute Grad-CAM mask.
    grayscale_cam = grad_cam(input_tensor=inputs)
    min_val = np.min(grayscale_cam)
    max_val = np.max(grayscale_cam)
    if max_val - min_val != 0:
        grayscale_cam = np.uint8(255 * (grayscale_cam - min_val) / (max_val - min_val))
    else:
        grayscale_cam = np.zeros_like(grayscale_cam, dtype=np.uint8)
    grayscale_cam = np.squeeze(grayscale_cam)

    # Overlay heatmap using OpenCV's COLORMAP_JET.
    colormap = cv2.COLORMAP_JET
    visualization_g = show_cam_on_image(inputs_np, grayscale_cam / 255, use_rgb=True, colormap=colormap)
    
    # Ensure the input image is RGB.
    if inputs_np.ndim == 2 or (inputs_np.ndim == 3 and inputs_np.shape[2] == 1):
        inputs_np = cv2.cvtColor(inputs_np, cv2.COLOR_GRAY2RGB)
    return {
        'cls_pred_bin': cls_pred_bin,
        'cls_pred_rounded': cls_pred_rounded,
        'inputs_np': inputs_np,
        'visualization_g': visualization_g
    }

def generate_gradcam_visualizations_test(model, layers, test_loader, DEVICE, threshold, save_dir):
    """
    Generate and save Grad-CAM visualizations for testing.
    """
    grad_cam = create_gradcam(model, layers, DEVICE)
    for i, data in enumerate(test_loader):
        inputs = data['image']
        dcm_name = data['dcm_name']

        outputs = process_gradcam_batch(model, inputs, grad_cam, DEVICE, threshold, rounding_precision=4)
        
        # Determine predicted class label.
        clspred = 'Pneumoperitoneum' if int(outputs['cls_pred_bin'].item()) == 1 else 'Non Pneumoperitoneum'
        
        plt.figure(figsize=(7,7), dpi=114.1)
        plt.imshow(outputs['visualization_g'])
        plt.axis('off')
        plt.title(f"{clspred}\n\nlikelihood:{outputs['cls_pred_rounded']}   thr:{threshold}", fontsize=18)
        
        file_name = f"{dcm_name[0].split('.')[0]}_gradcam.png"
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight', pad_inches=0.15)
        plt.close()
    print("Grad-CAM visualizations saved successfully!")
