import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from . import config as C

class VisualEncoder(nn.Module):
    """
    Visual Encoder using a pre-trained ResNet backbone to extract image embeddings.
    """
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        
        self.device = C.DEVICE
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            # ResNet18 output before fc is 512
            self.feature_dim = 512
        else:
            raise NotImplementedError(f"Model {model_name} not implemented yet")

        # Replace the final fully connected layer with Identity to get features
        # The average pooling layer output is what we want
        self.backbone.fc = nn.Identity()
        
        # If we want to project to a specific dimension different from backbone output
        if self.feature_dim != C.ENCODED_DIM:
            self.projection = nn.Linear(self.feature_dim, C.ENCODED_DIM)
        else:
            self.projection = nn.Identity()

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage() if not isinstance(Image.new('RGB', (1,1)), Image.Image) else transforms.Lambda(lambda x: x),
            transforms.Resize(C.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.to(self.device)
        self.eval()  # Default to eval mode

    def forward(self, x):
        """
        Forward pass for batch of images.
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, ENCODED_DIM)
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings

    def encode(self, image):
        """
        Encode a single image or batch of images.
        Args:
            image: PIL Image, numpy array (H, W, C), or list of them.
        Returns:
            torch.Tensor: Embedding vector(s) of shape (B, ENCODED_DIM) or (ENCODED_DIM,)
        """
        single_input = False
        if isinstance(image, (Image.Image, np.ndarray)):
            image = [image]
            single_input = True
            
        tensors = []
        for img in image:
            # Convert numpy to PIL if needed
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            # Apply transforms
            tensors.append(self.transform(img))
            
        # Stack into batch
        batch = torch.stack(tensors).to(self.device)
        
        # Inference
        with torch.no_grad():
            embeddings = self.forward(batch)
            
        if single_input:
            return embeddings[0]
        return embeddings
