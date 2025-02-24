
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR 

class TAFEModule(nn.Module):
    """
    Tumor-Aware Feature Encoding (TAFE) module.
    
    This module uses a SWIN-UNETR backbone to perform tumor segmentation
    and to extract multi-scale encoder features for classification. The segmentation
    branch produces segmentation logits (S ∈ R^(B×2×D×H×W)), while a global
    average pooling is applied to the encoder feature map to yield a compact 
    feature vector that is passed to a fully connected classification head.
    
    Parameters:
        backbone (nn.Module): Backbone segmentation network (default: SwinUNETR).
        img_size (tuple): Spatial dimensions of the input image.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for segmentation (e.g., 2 for background/tumor).
        feature_size (int): Base feature size for the backbone.
        depths (tuple): Depth (number of blocks) at each encoder stage.
        num_heads (tuple): Number of attention heads at each stage.
        classification_channels (int): Number of channels from the selected encoder features.
        num_classes (int): Number of classification output classes.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
        pretrained_path (str): Path to pretrained weights, if any.
        **kwargs: Additional arguments for the backbone.
    """
    def __init__(
        self,
        backbone: nn.Module = None,
        img_size=(96, 96, 96),
        in_channels=4,
        out_channels=2,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        classification_channels=None,
        num_classes=2,
        use_checkpoint=True,
        pretrained_path=None,
        **kwargs
    ):
        super().__init__()
        if backbone is None:
            self.backbone = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                depths=depths,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint,
                **kwargs
            )
        else:
            self.backbone = backbone

        # Determine classification channels
        if classification_channels is None:
            classification_channels = feature_size * 16

        # Classification head: map pooled features to final classes.
        self.classification_head = nn.Sequential(
            nn.Linear(classification_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        self.model_path = pretrained_path if pretrained_path is not None else ''

    def forward(self, x_in):
        """
        Forward pass.
        
        Args:
            x_in (tensor): Input tensor of shape [B, in_channels, D, H, W].
            
        Returns:
            seg_logits: Segmentation logits of shape [B, 2, D, H, W].
            cls_logits: Classification logits of shape [B, num_classes].
        """
        # Obtain hierarchical features from the backbone's Swin transformer encoder.
        hidden_states_out = self.backbone.swinViT(x_in, normalize=True)

        # --- Segmentation branch ---
        enc0 = self.backbone.encoder1(x_in)
        enc1 = self.backbone.encoder2(hidden_states_out[0])
        enc2 = self.backbone.encoder3(hidden_states_out[1])
        enc3 = self.backbone.encoder4(hidden_states_out[2])
        dec4 = self.backbone.encoder10(hidden_states_out[4])
        dec3 = self.backbone.decoder5(dec4, hidden_states_out[3])
        dec2 = self.backbone.decoder4(dec3, enc3)
        dec1 = self.backbone.decoder3(dec2, enc2)
        dec0 = self.backbone.decoder2(dec1, enc1)
        out = self.backbone.decoder1(dec0, enc0)
        seg_logits = self.backbone.out(out)  # [B, 2, D, H, W]

        # --- Classification branch ---
        x_deep = hidden_states_out[4]  # [B, classification_channels, D', H', W']
        x_pooled = F.adaptive_avg_pool3d(x_deep, (1, 1, 1)).view(x_in.size(0), -1)
        cls_logits = self.classification_head(x_pooled)

        return seg_logits, cls_logits
