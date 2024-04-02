import torch
from torch.nn import functional as F
# Distillation loss of cls and bbox reg
# https://github.com/dvlab-research/Dsig/tree/main/projects/Distillation
def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = torch.cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta

def kd_losses(self, gt_classes, gt_anchors_deltas, t_pred_class_logits, t_pred_anchor_deltas, pred_class_logits, pred_anchor_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.
        t_pred_class_logits, t_pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            t_pred_class_logits, t_pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = torch.stack(gt_classes)
        gt_anchors_deltas = torch.stack(gt_anchors_deltas)
        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        t_pred_class_logits = t_pred_class_logits[valid_idxs].softmax(dim=1)
        # kd logits loss
        T = 1
        loss_cls = F.kl_div(
            F.log_softmax(pred_class_logits[valid_idxs] / T, dim=1),
            t_pred_class_logits
        ) * (T * T)

        # kd regression loss
        loss = torch.abs(pred_anchor_deltas[foreground_idxs] - t_pred_anchor_deltas[foreground_idxs])
        loss = loss.sum()
        loss_box_reg = loss/max(1,num_foreground)

        return {"loss_kd_cls": loss_cls, "loss_kd_box_reg": loss_box_reg}
