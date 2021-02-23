from collections import defaultdict
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
    
    

    
def pixel_segementation_evaluation(target_labels: np.array, predicted_labels: np.array):
    """
    computes the f1 score, precision and recall by aggregating
    the contribution of each class. (class imbalance)
    """
    # target_labels    = target_labels.flatten()
    # predicted_labels = predicted_labels.flatten()
    F1               = f1_score(target_labels, predicted_labels, average='macro')
    P                = precision_score(target_labels, predicted_labels, average='macro')
    R                = recall_score(target_labels, predicted_labels, average='macro')

    return F1, P, R


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + 1e-6) / (union + 1e-6)  # We smooth our devision to avoid 0/0
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch


def IoU(target_labels: np.array, predicted_labels: np.array):
    target_labels = target_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    intersection = np.logical_and(target_labels, predicted_labels)
    union        = np.logical_or(target_labels, predicted_labels)
    iou_score    = (np.sum(intersection)+1e-6) / (np.sum(union)+1e-6)
    return iou_score


def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def confusion_matrix(target_labels: np.array, predicted_labels: np.array):
    target_labels = target_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    return confusion_matrix(target_labels, predicted_labels)
    

def accuracy(target_labels: np.array, predicted_labels: np.array):
    target_labels = target_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    return (np.sum(predicted_labels == target_labels) / target_labels.shape[0])




    
