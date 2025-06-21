# CMD Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

class T2FLAIR_3DFea(nn.Module):
    """
    3D CBAM-style attention on the T2-FLAIR difference,
    """
    def __init__(self, in_ch=1, base_ch=64, diff_amp=2.0, reduction=16):
        super().__init__()
        self.diff_amp = diff_amp
        self.base_ch = base_ch

        # initial 3D conv for each modality
        self.conv1 = nn.Conv3d(in_ch, base_ch,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        nn.init.kaiming_normal_(self.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        # ---- Channel Attention (shared MLP) ----
        self.mlp = nn.Sequential(
            nn.Conv3d(base_ch, base_ch // reduction,
                      kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch // reduction, base_ch,
                      kernel_size=1, bias=False)
        )

        # ---- Spatial Attention ----
        self.spatial_conv = nn.Conv3d(2, 1,
                                      kernel_size=3, stride=1, padding=1,
                                      bias=False)
        nn.init.kaiming_normal_(self.spatial_conv.weight,
                                mode='fan_out', nonlinearity='relu')

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t2, flair):
        # 1) Shared feature extraction
        feat_t2    = self.conv1(t2)
        feat_flair = self.conv1(flair)

        # 2) Amplify the pure difference
        diff_feat = (feat_t2 - feat_flair) * self.diff_amp

        # -------- Channel Attention --------
        # squeeze spatial dims → [B, C,1,1,1]
        avg_pool = F.adaptive_avg_pool3d(diff_feat, 1)
        max_pool = F.adaptive_max_pool3d(diff_feat, 1)
        
        # shared MLP → sum → sigmoid
        ca = self.mlp(avg_pool) + self.mlp(max_pool)
        ca = self.sigmoid(ca)  # [B, C,1,1,1]

        # apply channel attention
        feat_ca = diff_feat * ca  # broadcast over D,H,W

        # -------- Spatial Attention --------
        # pool along channel → [B,1,D,H,W] each
        avg_sp = torch.mean(feat_ca, dim=1, keepdim=True)
        max_sp = torch.max(feat_ca, dim=1, keepdim=True)[0]
        sp = torch.cat([avg_sp, max_sp], dim=1)              # [B,2,D,H,W]
        x = self.spatial_conv(sp)                            # [B,1,D,H,W]
        x = self.relu(x)
        sa = self.sigmoid(x)                                 # [B,1,D,H,W]

        # Combined attention map
        attn = ca * sa  # broadcast across C

        # -------- Residual gating --------
        feat_t2_aug = feat_t2 + attn * feat_t2
        feat_flair_aug = feat_flair + attn * feat_flair

        return feat_t2_aug, feat_flair_aug


class CMDModule(nn.Module):
    """
    Cross-Modality Differential (CMD) with CBAM-style attention.
    """
    def __init__(
        self,
        backbone: nn.Module = None,
        img_size=(96, 96, 96),
        in_channels=4,
        out_channels=4,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=2,
        use_checkpoint=True,
        pretrained_path=None,
        diff_amp=2.0,
        min_gate=0.1,
        base_ch=64,
        reduction=16,
        **kwargs
    ):
        super().__init__()
        # Segmentation backbone
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

        self.cmd_feature_extractor = T2FLAIR_3DFea(
            in_ch=1, base_ch=base_ch,
            diff_amp=diff_amp, reduction=reduction
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(base_ch * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        self.model_path = pretrained_path or ''

    def forward(self, x_in):
        bsz = x_in.size(0)

        # ---- Segmentation branch ----
        hidden = self.backbone.swinViT(x_in, normalize=True)
        enc0 = self.backbone.encoder1(x_in)
        enc1 = self.backbone.encoder2(hidden[0])
        enc2 = self.backbone.encoder3(hidden[1])
        enc3 = self.backbone.encoder4(hidden[2])
        dec4 = self.backbone.encoder10(hidden[4])
        dec3 = self.backbone.decoder5(dec4, hidden[3])
        dec2 = self.backbone.decoder4(dec3, enc3)
        dec1 = self.backbone.decoder3(dec2, enc2)
        dec0 = self.backbone.decoder2(dec1, enc1)
        out  = self.backbone.decoder1(dec0, enc0)
        seg_logits = self.backbone.out(out)  # [B,4,D,H,W]

        # ---- Whole-tumor gating mask ----
        seg_prob = torch.softmax(seg_logits, dim=1)
        tumor_prob = seg_prob[:,1:4,...].sum(dim=1,keepdim=True)
        gate = self.min_gate + (1.0 - self.min_gate) * tumor_prob

        # Extract & gate FLAIR/T2
        flair = x_in[:,0:1,...] * gate
        t2 = x_in[:,3:4,...] * gate

        # ---- CMD CBAM branch ----
        feat_t2_aug, feat_flair_aug = self.cmd_feature_extractor(t2, flair)

        # Global pooling & classification
        t2_pool = F.adaptive_avg_pool3d(feat_t2_aug, (1,1,1)).view(bsz, -1)
        flair_pool = F.adaptive_avg_pool3d(feat_flair_aug, (1,1,1)).view(bsz, -1)
        mismatch_feat = torch.cat([t2_pool, flair_pool], dim=1)
        cls_logits = self.classification_head(mismatch_feat)

        return seg_logits, cls_logits
