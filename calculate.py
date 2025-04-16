#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random 
from PIL import Image
import cv2 

def calculate_metrics(preds, labels):
    """计算 IoU, Dice, Precision, Recall 和 F1 分数"""
    preds_np = np.array(preds)  # 确保是 NumPy 数组
    labels_np = np.array(labels)  # 确保是 NumPy 数组

    # 计算 True Positive, False Positive, False Negative
    TP = np.sum((preds_np == 1) & (labels_np == 1))  # 真正例
    FP = np.sum((preds_np == 1) & (labels_np == 0))  # 假正例
    FN = np.sum((preds_np == 0) & (labels_np == 1))  # 假负例

    # Precision, Recall, F1
    precision = TP / (TP + FP + 1e-8)  # 加上1e-8避免除零
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # IoU
    intersection = np.sum((preds_np == 1) & (labels_np == 1))
    union = np.sum((preds_np == 1) | (labels_np == 1))
    iou = intersection / (union + 1e-8)  # 加上1e-8避免除零

    # Dice
    dice = 2 * intersection / (np.sum(preds_np) + np.sum(labels_np) + 1e-8)

    return iou, dice, precision, recall, f1



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # 使用sigmoid将logits转换为概率
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 平衡因子
        self.gamma = gamma  # 聚焦参数
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 将输入转换为概率
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算BCE损失
        bce = -(targets * torch.log(inputs + self.smooth) + 
                (1 - targets) * torch.log(1 - inputs + self.smooth))
        
        # 计算权重
        weights = torch.pow(1 - inputs, self.gamma) * targets + \
                 torch.pow(inputs, self.gamma) * (1 - targets)
        
        # 应用alpha平衡因子
        focal_loss = self.alpha * weights * bce
        
        return focal_loss.mean()
    
    
class CombinedLoss(nn.Module):
    def __init__(self, dice_smooth=1e-4, focal_alpha=0.25, focal_gamma=2.0, focal_smooth=1e-4):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, smooth=focal_smooth)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return 0.5 * dice + 0.5 * focal  # 各取50%权重组合
    
def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化，使实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True