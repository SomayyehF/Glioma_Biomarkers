
# CMD_Module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR 


# Helper modules for channel-wise pooling operations.
class ChannelMaxPooling3D(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.max_pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        out = self.max_pooling(x)
        out = out.permute(0, 4, 1, 2, 3)  # back to [B, C', D', H', W']
        return out

class ChannelAvgPooling3D(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.avg_pooling = nn.AvgPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        out = self.avg_pooling(x)
        out = out.permute(0, 4, 1, 2, 3)  # [B, C', D', H', W']
        return out

class T2FLAIR_3DFea(nn.Module):
    """
    3D Feature extractor for T2-FLAIR mismatch.
    
    This module computes features for the T2 and FLAIR MRI sequences using
    separate 3D convolutions, amplifies their difference by a factor γ,
    and then applies channel-wise max and average pooling to generate an
    attention map. This map is used to augment the original features.
    
    Parameters:
        in_ch (int): Number of input channels for each modality (default: 1).
        base_ch (int): Number of output channels for the initial convolution.
        diff_amp (float): Amplification factor (γ) for the difference.
    """
    def __init__(self, in_ch=1, base_ch=64, diff_amp=2.0):
        super().__init__()
        self.diff_amp = diff_amp
        
        # Initial convolutions for T2 and FLAIR
        self.conv1_t2 = nn.Conv3d(in_ch, base_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_flair = nn.Conv3d(in_ch, base_ch, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1_t2.weight)
        nn.init.kaiming_normal_(self.conv1_flair.weight)

        # Channel-wise pooling modules
        self.max_pool = ChannelMaxPooling3D(kernel_size=(1, 1, 64), stride=(1, 1, 64))
        self.avg_pool = ChannelAvgPooling3D(kernel_size=(1, 1, 64), stride=(1, 1, 64))

        # Attention map generation
        self.conv2 = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.relu2 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t2_3d, flair_3d):
        # Extract features from T2 and FLAIR volumes.
        feat_t2 = self.conv1_t2(t2_3d)     # [B, base_ch, D//2, H//2, W//2]
        feat_flair = self.conv1_flair(flair_3d)
        
        # Amplify the difference between modalities.
        diff_feat = (feat_t2 - feat_flair) * self.diff_amp
        
        # Channel-wise pooling to capture salient mismatch patterns.
        feat_max = self.max_pool(diff_feat)  # [B, 1, D', H', W']
        feat_avg = self.avg_pool(diff_feat)   # [B, 1, D', H', W']
        
        # Concatenate and generate an attention map.
        feat_cat = torch.cat((feat_max, feat_avg), dim=1)  # [B, 2, D', H', W']
        attn = self.conv2(feat_cat)
        attn = self.relu2(attn)
        attn_map = self.sigmoid(attn)  # [B, 1, D', H', W']
        
        # Apply the attention map to augment the original features.
        feat_t2_aug = feat_t2 + (attn_map * feat_t2)
        feat_flair_aug = feat_flair + (attn_map * feat_flair)
        
        return feat_t2_aug, feat_flair_aug

class CMDModule(nn.Module):
    """
    Cross-Modality Differential (CMD) module.
    
    This module combines tumor segmentation (via a backbone network) with 
    a dedicated 3D CMD branch to accentuate T2-FLAIR mismatches. It first 
    applies soft gating using the tumor probability map to the T2 and FLAIR 
    inputs, then extracts mismatch features, and finally performs classification.
    
    Parameters:
        backbone (nn.Module): SwinUNETR backbone.
        img_size, in_channels, out_channels, feature_size, depths, num_heads:
            Parameters for the backbone.
        num_classes (int): Number of classification output classes.
        use_checkpoint (bool): Whether to use checkpointing.
        pretrained_path (str): Path to pretrained weights.
        diff_amp (float): Amplification factor for the difference (γ).
        min_gate (float): Lower bound for soft gating (e.g., 0.1).
        base_ch (int): Number of output channels for the CMD convolutions.
        **kwargs: Additional backbone arguments.
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
        num_classes=2,
        use_checkpoint=True,
        pretrained_path=None,
        diff_amp=2.0,
        min_gate=0.1,
        base_ch=64,
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

        self.min_gate = min_gate
        self.cmd_feature_extractor = T2FLAIR_3DFea(in_ch=1, base_ch=base_ch, diff_amp=diff_amp)
        
        # Classification head for CMD branch (expects 2*base_ch features after pooling)
        self.classification_head = nn.Sequential(
            nn.Linear(base_ch * 2, 256),
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
        Args:
            x_in (tensor): Input tensor of shape [B, 4, D, H, W] with channels 
                           ordered as [FLAIR, T1, T1c, T2].
        
        Returns:
            seg_logits: Segmentation logits from the backbone.
            cls_logits: Classification logits computed from mismatch features.
        """
        bsz = x_in.size(0)
        # --- Segmentation branch for gating ---
        hidden_states_out = self.backbone.swinViT(x_in, normalize=True)
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
        
        # Compute tumor probability and apply soft gating.
        seg_prob = torch.softmax(seg_logits, dim=1)[:, 1:2, ...]  # tumor probability [B, 1, D, H, W]
        seg_prob_soft = self.min_gate + (1.0 - self.min_gate) * seg_prob
        
        # Extract FLAIR (channel 0) and T2 (channel 3) volumes.
        flair = x_in[:, 0:1, ...]
        t2 = x_in[:, 3:4, ...]
        gated_flair = flair * seg_prob_soft
        gated_t2 = t2 * seg_prob_soft
        
        # --- CMD branch for mismatch extraction ---
        feat_t2_aug, feat_flair_aug = self.cmd_feature_extractor(gated_t2, gated_flair)
        
        # Global pooling of each branch and concatenation.
        t2_pool = F.adaptive_avg_pool3d(feat_t2_aug, (1, 1, 1)).view(bsz, -1)
        flair_pool = F.adaptive_avg_pool3d(feat_flair_aug, (1, 1, 1)).view(bsz, -1)
        mismatch_feat = torch.cat([t2_pool, flair_pool], dim=1)  # [B, base_ch*2]
        
        cls_logits = self.classification_head(mismatch_feat)
        return seg_logits, cls_logits
