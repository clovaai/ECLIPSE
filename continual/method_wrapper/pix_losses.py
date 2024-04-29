import torch.nn as nn
import torch
from torch.nn import functional as F


def normalization(scores, mode="raw"):
    if mode == "raw":
        new_scores = scores
    elif mode == "prob":
        new_scores = F.softmax(scores, dim=1)
    else:
        raise NotImplementedError
    return new_scores


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):
        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        labels = targets.clone()  # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1., use_new=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.use_new = use_new

    def forward(self, inputs, targets, mask=None):
        """
        Inputs: a torch tensor of size BxCxHxW - the raw model scores
        Targets: a torch tensor of size BxC_oldxHxW - the old model scores
        """
        if self.use_new:
            outputs = torch.log_softmax(inputs, dim=1)
            outputs = outputs.narrow(1, 0, targets.shape[1])
        else:
            inputs = inputs.narrow(1, 0, targets.shape[1])
            outputs = torch.log_softmax(inputs, dim=1)

        labels = torch.softmax(targets * self.alpha, dim=1)
        loss = (outputs * labels).sum(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(
            inputs.device)

        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1),
                                      dim=1) - den  # B, H, W

        labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1))

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs