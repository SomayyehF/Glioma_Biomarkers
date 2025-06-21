
# DSF Module

class DSFModule(nn.Module):
    """
    Dual-Stream Fusion (DSF) module.
    
    This module fuses the classification outputs from the TAFE and CMD modules.
    It concatenates the two classification feature vectors and applies a lightweight
    multilayer perceptron (MLP) to produce the final classification logits.
    
    Parameters:
        input_dim_tafe (int): Dimensionality of the TAFE classification output.
        input_dim_cmd (int): Dimensionality of the CMD classification output.
        fusion_hidden_dim (int): Hidden dimension for the fusion MLP.
        num_classes (int): Number of output classes.
    """
    def __init__(self, input_dim_tafe, input_dim_cmd, fusion_hidden_dim=32, num_classes=2):
        super().__init__()
        fusion_dim = input_dim_tafe + input_dim_cmd
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_hidden_dim, num_classes)
        )
    
    def forward(self, c_tafe, c_cmd):
        """
        Args:
            c_tafe (tensor): Classification features from the TAFE module of shape [B, input_dim_tafe].
            c_cmd (tensor): Classification features from the CMD module of shape [B, input_dim_cmd].
        
        Returns:
            c_final: Final classification logits of shape [B, num_classes].
        """
        fused = torch.cat([c_tafe, c_cmd], dim=1)
        c_final = self.mlp(fused)
        return c_final

# Multi-Task Loss Function

def multi_task_loss(seg_logits, seg_gt, cls_logits, cls_labels, seg_loss_fn, cls_loss_fn, alpha=1.0, beta=1.0):
    """
    Computes the combined loss for the multitask network.
    
    Args:
        seg_logits (tensor): Segmentation logits [B, 2, D, H, W].
        seg_gt (tensor): Ground-truth tumor masks [B, 1, D, H, W].
        cls_logits (tensor): Classification logits [B, num_classes].
        cls_labels (tensor): Ground-truth labels [B].
        seg_loss_fn: Loss function for segmentation (e.g., Dice loss).
        cls_loss_fn: Loss function for classification (e.g., CrossEntropyLoss).
        alpha (float): Weight for segmentation loss.
        beta (float): Weight for classification loss.
    
    Returns:
        combined_loss: Weighted sum of segmentation and classification losses.
        seg_loss: Segmentation loss.
        cls_loss: Classification loss.
    """
    seg_loss = seg_loss_fn(seg_logits, seg_gt)
    cls_loss = cls_loss_fn(cls_logits, cls_labels)
    combined_loss = alpha * seg_loss + beta * cls_loss
    return combined_loss, seg_loss, cls_loss
