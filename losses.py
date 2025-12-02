import torch
import torch.nn as nn
import torch.nn.functional as F


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    Multi-label categorical cross-entropy.
    Works for PyTorch 2.x + Python 3.11.
    """
    # y_true = (0/1), y_pred = logits (not sigmoid)
    y_pred = (1 - 2 * y_true) * y_pred

    # mask positive & negative logits
    y_pred_neg = y_pred - y_true * 1e30     # mask positive
    y_pred_pos = y_pred - (1 - y_true) * 1e30  # mask negative

    zeros = torch.zeros_like(y_pred[..., :1])

    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

    # FIX axis -> dim
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return neg_loss + pos_loss


class balanced_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        loss = multilabel_categorical_crossentropy(labels, logits)
        return loss.mean()

    def get_label(self, logits, num_labels=-1):
        th_logit = torch.zeros_like(logits[..., :1])
        output = torch.zeros_like(logits)

        mask = logits > th_logit

        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask

        output[mask] = 1.0

        # ensure class 0 = NA class
        output[:, 0] = (output[:, 1:].sum(dim=1) == 0).to(logits.dtype)
        return output


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # clone to avoid modifying original labels
        labels = labels.clone()

        # TH label
        th_label = torch.zeros_like(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(dim=1)

        # Rank negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(dim=1)

        loss = (loss1 + loss2).mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        # threshold = logit of NA class (class 0)
        th_logit = logits[:, 0].unsqueeze(-1)
        output = torch.zeros_like(logits)

        mask = logits > th_logit

        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask

        output[mask] = 1.0

        # ensure at least NA class if no positive predicted
        output[:, 0] = (output.sum(dim=1) == 0).to(logits.dtype)
        return output
