import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ModelResNet(nn.Module):

    def __init__(self):
        super(ModelResNet, self).__init__()

        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        
    def forward(self, input):
        
        # Get the BERT output
        output = self.resnet50(input)

        return output
    

