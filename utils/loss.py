import torch 
from torch.nn import functional as F

def soft_jaccard(outputs, targets):
    eps = 1e-15
    jaccard_target = (targets == 1).float()
    jaccard_output = F.sigmoid(outputs)
    intersection   = (jaccard_output * jaccard_target).sum()
    union          = jaccard_output.sum() + jaccard_target.sum()
    return intersection / (union - intersection + eps)


def generalised_loss(output, target, weights=0.7):
    """
    generalised loss
    http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers
    /w4/Iglovikov_TernausNetV2_Fully_Convolutional_CVPR_2018_paper.pdf
    
    By minimizing this loss function, we simultaneously maxi-mize predicted
    probabilities for the right class for each pixeland maximize the intersection
    over unionJbetween masksand corresponding predictions
    """
    binary_cross_entropy = F.binary_cross_entropy_with_logits(output, target)
    output               = F.sigmoid(output)
    jaccard_index        = soft_jaccard(output, target)
    print('JI', jaccard_index.item())
    print('BE',binary_cross_entropy.item())
    loss = binary_cross_entropy * weights  -  (1 - jaccard_index) * (1 - weights)
    return loss
    
def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, min=1e-7, max = 1-1e-7)
    if weights is not None:
        assert len(weights) == 2
        log_1 = torch.log(output)
        log_0 = torch.log(1 - output)
        loss = weights[1] * (target * log_1) + weights[0] * ((1 - target) * log_0)
    else:
        log_1 = torch.log(output)
        log_0 = torch.log(1 - output)
        loss = target * log_1 + (1 - target) * log_0
    return torch.neg(torch.mean(loss))
