import segmentation_models_pytorch as smp
import torch.nn as nn

import segmentation_models_pytorch as smp
import torch.nn as nn

class LateralClassificationModel(nn.Module):
    """
    A classification model that uses only the encoder (lateral image) from an SMP architecture.
    The bottleneck features are pooled and fed through a fully connected layer.
    """
    def __init__(self, layers):
        super().__init__()
        # For MIT encoders, the input channels must be 3.
        self.is_mit_encoder = 'mit' in layers
        in_channels = 3 if self.is_mit_encoder else 1
        # Get the encoder from SMP (no decoder, no segmentation head)
        self.encoder = smp.encoders.get_encoder(
            name=layers,
            weights='imagenet',
            in_channels=in_channels,
            depth=5
        )
        # Create classification head: global average pooling, flatten, and linear layer.
        bottleneck_channels = self.encoder.out_channels[-1]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(bottleneck_channels, 1)
    
    def forward(self, x):
        if self.is_mit_encoder and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # Get list of feature maps; use the last one as the bottleneck.
        features = self.encoder(x)
        bottleneck = features[-1]
        pooled = self.avg_pool(bottleneck)
        flattened = self.flatten(pooled)
        logits = self.fc(flattened)
        return logits


# class LateralClassificationModel(nn.Module):
#     def __init__(self, layers, aux_params=None):
#         super().__init__()
#         # For lateral images, we use 1 input channel (or 3 for mit encoders)
#         self.is_mit_encoder = 'mit' in layers
#         in_channels = 3 if self.is_mit_encoder else 1
#         # Create a U-Net from SMP; we only need the encoder part.
#         base_model = smp.Unet(layers, encoder_weights='imagenet', in_channels=in_channels, classes=1, aux_params=None)
#         self.encoder = base_model.encoder
#         # Use the encoder’s final output channels for the classifier head.
#         out_channels = self.encoder.out_channels[-1]
#         self.cls_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(out_channels, 1)
#         )
        
#     def forward(self, x):
#         # For non-mit encoders with single channel images, replicate to 3 channels.
#         if not self.is_mit_encoder and x.size(1) == 1:
#             x = x.repeat(1, 3, 1, 1)
#         features = self.encoder(x)
#         bottleneck = features[-1]
#         cls_logits = self.cls_head(bottleneck)
#         return cls_logits
        

