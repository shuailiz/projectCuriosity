import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image
import numpy as np
from . import config as C

class VisualEncoder(nn.Module):
    """
    Visual Encoder using a pre-trained ResNet backbone to extract image embeddings.
    """
    def __init__(self, model_name=None, pretrained=True):
        super().__init__()
        
        self.device = C.DEVICE
        
        if model_name is None:
            model_name = C.ENCODER_MODEL
        
        # Load pre-trained model
        resnet_configs = {
            'resnet18': (models.resnet18, 512),
            'resnet34': (models.resnet34, 512),
            'resnet50': (models.resnet50, 2048),
        }
        
        if model_name not in resnet_configs:
            raise ValueError(f"Unsupported model '{model_name}'. Choose from: {list(resnet_configs.keys())}")
        
        model_fn, self.feature_dim = resnet_configs[model_name]
        self.backbone = model_fn(pretrained=pretrained)
        print(f"VisualEncoder: {model_name} (feature_dim={self.feature_dim} -> {C.ENCODED_DIM})")

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


class VideoEncoder(nn.Module):
    """
    Video Encoder using a pretrained R3D-18 (3D ResNet) backbone.

    Encodes a short clip of CLIP_FRAMES frames into a single embedding vector,
    capturing temporal motion information across the clip.

    Input clip format: list of CLIP_FRAMES numpy arrays (H, W, 3) RGB.
    If fewer than CLIP_FRAMES frames are provided, the last frame is repeated
    to pad to the required length.
    """

    def __init__(self):
        super().__init__()
        self.device = C.DEVICE
        self.clip_frames = C.CLIP_FRAMES

        # Load pretrained R3D-18
        weights = R3D_18_Weights.DEFAULT
        backbone = r3d_18(weights=weights)
        print(f"VideoEncoder: r3d_18 pretrained (feature_dim=512 -> {C.ENCODED_DIM})")

        # Remove the classification head; keep up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 512, 1, 1, 1)

        if 512 != C.ENCODED_DIM:
            self.projection = nn.Linear(512, C.ENCODED_DIM)
        else:
            self.projection = nn.Identity()

        # R3D-18 uses the same ImageNet normalization as ResNet
        self.frame_transform = transforms.Compose([
            transforms.Resize(C.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989]),
        ])

        self.to(self.device)
        self.eval()

    def _prepare_clip(self, frames):
        """
        Convert a list of numpy RGB frames into an R3D-18 input tensor.

        Args:
            frames: list of numpy arrays (H, W, 3) uint8 RGB, length >= 1.
        Returns:
            Tensor (1, C, T, H, W) on self.device  — batch of 1 clip.
        """
        # Pad to CLIP_FRAMES if needed
        while len(frames) < self.clip_frames:
            frames = frames + [frames[-1]]
        frames = frames[:self.clip_frames]

        # Each frame: (H, W, 3) uint8 -> PIL -> transform -> (3, H, W)
        tensors = []
        for f in frames:
            img = Image.fromarray(f) if isinstance(f, np.ndarray) else f
            tensors.append(self.frame_transform(img))  # (3, H, W)

        # Stack to (T, C, H, W) then permute to (C, T, H, W)
        clip = torch.stack(tensors, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
        return clip.unsqueeze(0).to(self.device)  # (1, C, T, H, W)

    def forward(self, x):
        """
        Args:
            x: Tensor (B, C, T, H, W)
        Returns:
            Tensor (B, ENCODED_DIM)
        """
        features = self.backbone(x)          # (B, 512, 1, 1, 1)
        features = features.flatten(1)       # (B, 512)
        return self.projection(features)     # (B, ENCODED_DIM)

    def encode(self, frames):
        """
        Encode a clip of frames into a single embedding.

        Args:
            frames: list of numpy arrays (H, W, 3) RGB. Length should be ~CLIP_FRAMES.
        Returns:
            Tensor (ENCODED_DIM,) on CPU — detached.
        """
        clip = self._prepare_clip(frames)
        with torch.no_grad():
            emb = self.forward(clip)   # (1, ENCODED_DIM)
        return emb.squeeze(0)          # (ENCODED_DIM,)
