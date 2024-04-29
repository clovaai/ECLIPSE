import torch
from torch.nn import functional as F

def sigmoid_cross_entropy(inputs, targetsreduction='mean', ignore_index=255):
    one_hot_targets = F.one_hot(targets.long(), inputs.shape[1]).permute(0, 2, 1).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, one_hot_targets, reduction=reduction)
    
    return loss


def sigmoid_focal_loss(inputs, targets, alpha=10, gamma=2, reduction='mean', ignore_index=255):
    one_hot_targets = F.one_hot(targets.long(), inputs.shape[1]).permute(0, 2, 1).float()
    
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, one_hot_targets, reduction="none")
    p_t = p * one_hot_targets + (1 - p) * (1 - one_hot_targets)
    loss = alpha * ce_loss * ((1 - p_t) ** gamma)

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
            
    elif reduction == "sum":
        loss = loss.sum()

    return loss * 50

def focal_loss(inputs, targets, alpha=10, gamma=2, reduction='mean', ignore_index=255):
    ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    f_loss = alpha * (1 - pt) ** gamma * ce_loss
    if reduction == 'mean':
        f_loss = f_loss.mean()
    elif reduction == 'wmean':
        f_loss = f_loss.sum() / ((1 - pt) ** gamma).sum()
    return f_loss


def focal_uce_loss(inputs, targets, old_cl, alpha=10, gamma=2, reduction='mean'):
    ce_loss = unbiased_cross_entropy_loss(inputs, targets, reduction="none", old_cl=old_cl)
    pt = torch.exp(-ce_loss)
    f_loss = alpha * (1 - pt) ** gamma * ce_loss
    if reduction == 'mean':
        f_loss = f_loss.mean()
    elif reduction == 'wmean':
        f_loss = f_loss.sum() / ((1 - pt) ** gamma).sum()
    return f_loss


def unbiased_cross_entropy_loss(inputs, targets, old_cl, weights=None, reduction='mean'):
    outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
    den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax

    to_sum = torch.cat((inputs[:, -1:], inputs[:, 0:old_cl]), dim=1)
    outputs[:, -1] = torch.logsumexp(to_sum, dim=1) - den  # B, K       p(O)
    outputs[:, :-1] = inputs[:, :-1] - den.unsqueeze(dim=1)  # B, N, K    p(N_i)

    loss = F.nll_loss(outputs, targets, weight=weights, reduction=reduction)

    return loss


def focal_distillation_loss(inputs, targets, use_new=False, alpha=1, gamma=2, ):
    if use_new:
        outputs = torch.log_softmax(inputs, dim=1)  # remove no-class
        outputs = torch.cat((outputs[:, :targets.shape[1] - 1], outputs[:, -1:]), dim=1)  # only old classes or EOS
    else:
        inputs = torch.cat((inputs[:, :targets.shape[1] - 1], inputs[:, -1:]), dim=1)  # only old classes or EOS
        outputs = torch.log_softmax(inputs, dim=1)
    labels = torch.softmax(targets * alpha, dim=1)
    labels_log = torch.log_softmax(targets * alpha, dim=1)

    loss = (labels * (labels_log-outputs)).sum(dim=1)
    pt = torch.exp(torch.clamp(-loss, max=0))
    f_loss = alpha * (1 - pt) ** gamma * loss

    return f_loss.mean()


def L2_distillation_loss(inputs, targets, use_new=False):
    inputs = torch.cat((inputs[:, :targets.shape[1]-1], inputs[:, -1:]), dim=1)  # only old classes or EOS

    labels = torch.softmax(targets, dim=1)
    outputs = torch.softmax(inputs, dim=1)

    # keep only the informative ones -> The not no-obj masks
    # keep = (labels.argmax(dim=1) != targets.shape[1]-1)  # B x Q

    loss = torch.pow((outputs - labels), 2).sum(dim=1)
    # loss = (loss * keep).sum() / (keep.sum() + 1e-4)  # keep only obj queries, 1e-4 to avoid NaN
    return loss.mean()


def knowledge_distillation_loss(inputs, targets, reweight=False, gamma=2., temperature=1., use_new=True):
    if use_new:
        outputs = torch.log_softmax(inputs, dim=1)  # remove no-class
        outputs = torch.cat((outputs[:, :targets.shape[1]-1], outputs[:, -1:]), dim=1)  # only old classes or EOS
    else:
        inputs = torch.cat((inputs[:, :targets.shape[1]-1], inputs[:, -1:]), dim=1)  # only old classes or EOS
        outputs = torch.log_softmax(inputs, dim=1)
    labels = torch.softmax(targets * temperature, dim=1)
    labels_log = torch.log_softmax(targets * temperature, dim=1)

    loss = (labels*(labels_log - outputs)).sum(dim=1)  # B x Q
    # Re-weight no-cls queries as in classification
    if reweight:
        loss = ((1-labels[:, -1]) ** gamma * loss).sum() / ((1-labels[:, -1]) ** gamma).sum()
    else:
        loss = loss.mean()
    return loss


def unbiased_knowledge_distillation_loss(inputs, targets, reweight=False, gamma=2., temperature=1.):
    targets = targets * temperature

    den = torch.logsumexp(inputs, dim=1)  # B, C
    outputs_no_bgk = inputs[:, :targets.shape[1]-1] - den.unsqueeze(dim=1)  # B, OLD_CL, Q
    outputs_bkg = torch.logsumexp(inputs[:, targets.shape[1]-1:], dim=1) - den  # B, Q
    labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, Q
    labels_soft = torch.log_softmax(targets, dim=1)

    loss = labels[:, -1] * (labels_soft[:, -1] - outputs_bkg) + \
           (labels[:, :-1] * (labels_soft[:, :-1] - outputs_no_bgk)).sum(dim=1)  # B, Q
    # Re-weight no-cls queries as in classificaton
    if reweight:
        loss = ((1-labels[:, -1]) ** gamma * loss).sum() / ((1-labels[:, -1]) ** gamma).sum()
    else:
        loss = loss.mean()
    return loss
