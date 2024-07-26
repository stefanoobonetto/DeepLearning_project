import cv2
import numpy as np
import torch
from PIL import Image
from functions import *

# Check for the appropriate device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def create_gradcam(image, model, target_layers):
    """
    Creates a Grad-CAM heatmap for the given image and model.
    
    Args:
        image (numpy array): The input image.
        model (torch.nn.Module): The model to be used for generating Grad-CAM.
        target_layers (list): The target layers for Grad-CAM.
        
    Returns:
        PIL.Image: The Grad-CAM heatmap image.
        tuple: The centroid of the most important region.
        numpy array: The important region identified by Grad-CAM.
    """
    # Preprocess the image and create input tensor
    input_tensor = transform(Image.fromarray((image * 255).astype(np.uint8))).unsqueeze(0).to(device)
    
    # Initialize Grad-CAM
    cam_algorithm = GradCam(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale_cam = cam_algorithm(input_tensor=input_tensor)[0]
    heatmap = (np.uint8(255 * grayscale_cam))

    # Identify the most important region
    threshold_value = np.max(heatmap)
    important_region = np.where(heatmap == threshold_value)
    
    # Calculate the centroid of the important region
    centroid_x = important_region[1][0]
    centroid_y = important_region[0][0]
    centroid = (int(centroid_x), int(centroid_y))

    # Overlay the heatmap on the original image
    cam_image = show_cam_on_image(image, grayscale_cam)

    return Image.fromarray(cam_image), centroid, important_region

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET, image_weight: float = 0.5):
    """
    Overlay the heatmap on the image.
    
    Args:
        img (numpy array): The original image.
        mask (numpy array): The Grad-CAM heatmap.
        colormap (int): The colormap to apply to the heatmap.
        image_weight (float): Weight for the original image in the overlay.
        
    Returns:
        numpy array: The combined image with heatmap overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def reshape_transform(tensor, height=14, width=14):
    """
    Reshape and transform the tensor for visualization.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        height (int): The height to reshape to.
        width (int): The width to reshape to.
        
    Returns:
        torch.Tensor: The reshaped tensor.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform=None):
        """
        Initialize hooks to save activations and gradients.
        
        Args:
            model (torch.nn.Module): The model to hook.
            target_layers (list): The layers to hook.
            reshape_transform (callable): Function to reshape the activations and gradients.
        """
        self.model = model.eval()
        self.target_layers = target_layers
        self.activations = []
        self.gradients = []
        self.handles = []
        self.reshape_transform = reshape_transform
        for target_layer in self.target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_full_backward_hook(self.save_gradient))
        
    def save_activation(self, module, input, output):
        """
        Save the activation of the forward pass.
        
        Args:
            module (torch.nn.Module): The module hooked.
            input (torch.Tensor): The input tensor.
            output (torch.Tensor): The output tensor.
        """
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        Save the gradient of the backward pass.
        
        Args:
            module (torch.nn.Module): The module hooked.
            grad_input (torch.Tensor): The input gradient tensor.
            grad_output (torch.Tensor): The output gradient tensor.
        """
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients
    
    def __call__(self, x):
        """
        Make the model callable to retrieve activations and gradients.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The model output.
        """
        self.gradients = []
        self.activations = []
        return self.model(x)
    
    def release(self):
        """
        Remove all hooks.
        """
        for handle in self.handles:
            handle.remove()

class GradCam:
    def __init__(self, model, target_layers, reshape_transform=None):
        """
        Initialize the Grad-CAM object.
        
        Args:
            model (torch.nn.Module): The model to use for Grad-CAM.
            target_layers (list): The layers to hook for Grad-CAM.
            reshape_transform (callable): Function to reshape the activations and gradients.
        """
        self.model = model.eval()
        self.target_layers = target_layers
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)
        self.reshape_transform = reshape_transform

    def forward(self, input_tensor, targets=None):
        """
        Perform the forward pass and compute Grad-CAM.
        
        Args:
            input_tensor (torch.Tensor): The input tensor.
            targets (torch.Tensor, optional): The target tensor.
        
        Returns:
            numpy array: The computed Grad-CAM heatmap.
        """
        output = self.activations_and_grads(input_tensor)
        output = output[:, indices_in_1k]
        target_category = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        loss = output[:, target_category].sum()
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, target_category)
        return self.aggregate_multi_layers(cam_per_layer)

    def aggregate_multi_layers(self, cam_per_target_layer):
        """
        Aggregate the Grad-CAM heatmaps from multiple layers.
        
        Args:
            cam_per_target_layer (list): List of heatmaps from each layer.
        
        Returns:
            numpy array: The aggregated heatmap.
        """
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result, target_size=(224, 224))
    
    def scale_cam_image(self, cam, target_size=None):
        """
        Scale the Grad-CAM heatmap to the target size.
        
        Args:
            cam (numpy array): The Grad-CAM heatmap.
            target_size (tuple, optional): The target size to scale to.
        
        Returns:
            numpy array: The scaled heatmap.
        """
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-7)
        if target_size:
            cam = [cv2.resize(np.float32(img), target_size) for img in cam]
        return np.float32(cam)
    
    def get_cam_image(self, activations, grads):
        """
        Compute the Grad-CAM image from activations and gradients.
        
        Args:
            activations (numpy array): The activations.
            grads (numpy array): The gradients.
        
        Returns:
            numpy array: The computed Grad-CAM image.
        """
        weights = np.mean(grads, axis=(2, 3))
        weighted_activations = weights[:, :, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    def compute_cam_per_layer(self, input_tensor, target_category):
        """
        Compute the Grad-CAM heatmap for each layer.
        
        Args:
            input_tensor (torch.Tensor): The input tensor.
            target_category (int): The target category for Grad-CAM.
        
        Returns:
            list: List of Grad-CAM heatmaps for each layer.
        """
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = (input_tensor.size(2), input_tensor.size(3))

        cam_per_target_layer = []
        for i, target_layer in enumerate(self.target_layers):
            layer_activations = activations_list[i] if i < len(activations_list) else None
            layer_grads = grads_list[i] if i < len(grads_list) else None

            cam = self.get_cam_image(layer_activations, layer_grads)
            cam = np.maximum(cam, 0)
            cam_per_target_layer.append(self.scale_cam_image(cam, target_size)[:, None, :])

        return cam_per_target_layer

    def __call__(self, input_tensor, targets=None, aug_smooth=False):
        """
        Make the Grad-CAM object callable.
        
        Args:
            input_tensor (torch.Tensor): The input tensor.
            targets (torch.Tensor, optional): The target tensor.
            aug_smooth (bool, optional): Flag for augmentation smoothing.
        
        Returns:
            numpy array: The computed Grad-CAM heatmap.
        """
        return self.forward(input_tensor, targets)
    
    def __del__(self):
        """
        Destructor to release hooks.
        """
        self.activations_and_grads.release()
