import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


# -------------------------
# CBAM Module
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        out = torch.sigmoid(avg_out + max_out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv1(x_cat))
        return out * x


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# -------------------------
# CORAL Head
# -------------------------
class CORALHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP layer
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        # CORAL
        self.shared = nn.Linear(512, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        x = self.gap(x)         # B, C, 1, 1
        x = torch.flatten(x, 1) # B, C
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)

        s = self.shared(x)            # (B, 1)
        out = s + self.biases
        return out


def init_coral_biases(head: CORALHead, class_counts):
    device = head.biases.device   # ensure we put biases on same device as model
    counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
    cum_le = torch.cumsum(counts, 0)[:-1]
    cum_gt = counts.sum() - cum_le
    probs = (cum_gt + 1e-6) / (cum_gt + cum_le + 1e-6)
    head.biases.data = torch.log(probs / (1 - probs)).to(device)



# -------------------------
# Full Model (ResNet50 + CBAM + CORAL Head)
# -------------------------
class AgePredictionCORAL(nn.Module):
    def __init__(self, num_classes=44):
        super().__init__()
        backbone = models.resnet50(pretrained=True)

        # Keep backbone conv layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        # Add CBAM after each layer
        self.layer1 = nn.Sequential(backbone.layer1, CBAM(256))
        self.layer2 = nn.Sequential(backbone.layer2, CBAM(512))
        self.layer3 = nn.Sequential(backbone.layer3, CBAM(1024))
        self.layer4 = nn.Sequential(backbone.layer4, CBAM(2048))

        in_features = backbone.fc.in_features
        self.coral = CORALHead(in_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.coral(x)  # B, K-1
        return logits


# -------------------------
# CORAL Loss
# -------------------------
def coral_loss_v0(logits, labels, num_classes, device, pos_weights=None):
    B, K_minus1 = logits.shape
    ord_targets = torch.zeros((B, K_minus1), device=device)
    for i in range(B):
        ord_targets[i, :labels[i]] = 1
    if pos_weights is not None:
        pos_weights = torch.tensor(pos_weights, device=device)
    loss = F.binary_cross_entropy_with_logits(
        logits, ord_targets, pos_weight=pos_weights, reduction="mean"
    )
    return loss


def coral_loss(logits, labels, num_classes, device, pos_weights=None, smoothing=0.1):
    B, K_minus1 = logits.shape
    ord_targets = torch.zeros((B, K_minus1), device=device)
    for i in range(B):
        ord_targets[i, :labels[i]] = 1
    if smoothing > 0:
        ord_targets = ord_targets * (1 - smoothing) + 0.5 * smoothing
    loss = F.binary_cross_entropy_with_logits(
        logits,
        ord_targets,
        pos_weight=None if pos_weights is None else torch.tensor(pos_weights, device=device),
        reduction="mean"
    )
    return loss


# -------------------------
# Mixup
# -------------------------
def mixup_data(x, y, alpha=0.4, max_delta=5):
    B = y.size(0)
    perm = torch.randperm(B, device=x.device)
    mask = (y - y[perm]).abs() <= max_delta
    perm = torch.where(mask, perm, torch.arange(B, device=x.device))
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x + (1 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


def mixup_coral_loss(logits, y_a, y_b, lam, num_classes, device):
    return lam * coral_loss(logits, y_a, num_classes, device) + \
           (1 - lam) * coral_loss(logits, y_b, num_classes, device)


# -------------------------
# Decode predictions
# -------------------------
def class_to_age(class_name: str) -> int:
    if class_name == "below_8":
        return 7
    elif class_name == "above_50":
        return 51
    else:
        return int(class_name)
    
idx_to_class = {
    0: '008', 1: '009', 2: '010', 3: '011', 4: '012', 5: '013', 6: '014',
    7: '015', 8: '016', 9: '017', 10: '018', 11: '019', 12: '20', 13: '21',
    14: '22', 15: '23', 16: '24', 17: '25', 18: '26', 19: '27', 20: '28',
    21: '29', 22: '30', 23: '31', 24: '32', 25: '33', 26: '34', 27: '35',
    28: '36', 29: '37', 30: '38', 31: '39', 32: '40', 33: '41', 34: '42',
    35: '43', 36: '44', 37: '45', 38: '46', 39: '47', 40: '48', 41: '49',
    42: '50', 43: 'above_50', 44: 'below_8'
}

def coral_decode(logits, threshold=0.5, idx_to_class=idx_to_class):
    probs = torch.sigmoid(logits) 
    pred_idx = torch.sum(probs > threshold, dim=1)  

    if idx_to_class:
        pred_ages = [class_to_age(idx_to_class[i.item()]) for i in pred_idx]
        return pred_ages
    return pred_idx