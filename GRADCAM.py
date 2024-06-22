import cv2
import torch
import numpy as np
from torchvision import models
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'mps'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transformGradCam = T.Compose([                                                  # Resize each PIL image to 224 x 224
    T.ToTensor(),                                                           # Convert Numpy to Pytorch Tensor
    T.Normalize(mean, std)                                                  # Normalize with ImageNet mean
])

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET, image_weight: float = 0.5):
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

#### TEMPORARY ####
class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

class ActivationsAndGradients:
    def __init__(self, model, target_layers):
        self.model = model.eval()
        self.target_layers = target_layers
        self.activations = []
        self.gradients = []
        self.handles = []
        for target_layer in self.target_layers:
            self.handles.append(target_layer.register_forward_hook(self.saveActivation))
            self.handles.append(target_layer.register_full_backward_hook(self.saveGradient))
        
    def saveActivation(self, module, input, output):
        self.activations.append(output.detach())
    
    def saveGradient(self, module, input, output):
        # Gradient data is captured in the backward phase
        self.gradients.append(output[0].detach())
    
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

def get_2d_projection(activation_batch):
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = activations.reshape(activations.shape[0], -1).T
        reshaped_activations -= reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)

class GradCam:
    def __init__(self, model, target_layers):
        self.model = model.eval()
        self.target_layers = target_layers
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers)

    
    def forward(self, input_tensor, targets = None):
        output = self.activations_and_grads(input_tensor)
        targets = targets or [ClassifierOutputTarget(category) for category in np.argmax(output.cpu().data.numpy(), axis=-1)]

        self.model.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, output)])
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets)
        return self.aggregate_multi_layers(cam_per_layer)

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)
    
    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))
        else:
            raise ValueError("Invalid grads shape. Shape of grads should be 4 (2D image) or 5 (3D image).")

    def scale_cam_image(self, cam, target_size=None):
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-7)
        if target_size:
            cam = [cv2.resize(np.float32(img), target_size) for img in cam]
        return np.float32(cam)
    
    def get_cam_image(self, input_tensor, target_layer, targets, activations, grads, eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations if len(activations.shape) == 4 else weights[:, :, None, None, None] * activations
        cam = get_2d_projection(weighted_activations) if eigen_smooth else weighted_activations.sum(axis=1)
        return cam


    def compute_cam_per_layer(self, input_tensor, targets):
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = (input_tensor.size(2), input_tensor.size(3))

        cam_per_target_layer = []
        for i, target_layer in enumerate(self.target_layers):
            layer_activations = activations_list[i] if i < len(activations_list) else None
            layer_grads = grads_list[i] if i < len(grads_list) else None

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads)
            cam = np.maximum(cam, 0)
            cam_per_target_layer.append(self.scale_cam_image(cam, target_size)[:, None, :])

        return cam_per_target_layer

    def __call__(self, input_tensor, targets=None, aug_smooth=False):
        return self.forward(input_tensor, targets)



if __name__ == '__main__':

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    target_layers = [model.layer4]
    path_img = "/Users/simoneroman/Desktop/DL/old/Project/datasets/imagenet-a/n01498041/0.000116_digital clock _ digital clock_0.865662.jpg"  # Ensure this path is correct

    rgb_img = cv2.imread(path_img)


    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = transformGradCam(rgb_img).unsqueeze(0).to(device)   #preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
    
    cam_algorithm = GradCam(model = model, target_layers = target_layers)
    grayscale_cam = cam_algorithm(input_tensor=input_tensor)[0]
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("GradCam", cam_image)
    cv2.waitKey(0)