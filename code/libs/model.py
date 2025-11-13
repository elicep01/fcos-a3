import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn

import torch.nn.functional as F

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """

        cls_logits = []

        for tensor in x:
            conv_output = self.conv(tensor)
            cls_logits.append(self.cls_logits(conv_output))

        return cls_logits


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """

        reg_outputs = []
        ctr_logits = []

        for tensor in x:
            conv_output = self.conv(tensor)
            reg_outputs.append(self.bbox_reg(conv_output))
            ctr_logits.append(self.bbox_ctrness(conv_output))

        return reg_outputs, ctr_logits


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function depends on if the model is in training
    or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detection results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)
    * You might want to double check the format of 2D coordinates saved in points

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """
    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        """
        training pipeline
        1) make targets for every location across all FPN levels
            - labels: 0 = background, 1..C = class id
            - reg: distances [l,t,r,b] / stride
            - ctr: centerness in (0,1)
        2) flatten model outputs to match those targets
        3) compute 3 losses:
            - classification (focal) on ALL locations
            - regression (GIoU) on POSITIVES only
            - centerness (BCE-with-logits) on POSITIVES only
        each loss is divided by #positives for stability.
        """

        DEBUG = False  # set True to see shapes/positives once

        device = cls_logits[0].device
        N = len(targets)
        L = len(points)
        C = self.num_classes  # do not hard-code 

        # helpers 
        def _flatten_levels(level_list):
            # [N, ch, H, W] -> [N, H*W, ch], then concat levels -> [N, S, ch]
            return torch.cat(
                [t.permute(0, 2, 3, 1).reshape(t.size(0), -1, t.size(1)) for t in level_list],
                dim=1
            )

        # points per level: [HW,2], stitching offsets to build a single [S] axis
        pts_per_level = [p.view(-1, 2).to(device) for p in points]  # each (x,y)
        num_loc = [p.shape[0] for p in pts_per_level]               # HW per level
        offsets, acc = [], 0
        for n_ in num_loc:
            offsets.append(acc)
            acc += n_
        S = acc  # total locations across all levels

        # target tensors 
        labels_all   = torch.zeros((N, S),    dtype=torch.long,    device=device)  # 0 = background
        reg_tgts_all = torch.zeros((N, S, 4), dtype=torch.float32, device=device)  # [l,t,r,b]/stride
        ctr_tgts_all = torch.zeros((N, S),    dtype=torch.float32, device=device)  # (0,1)
        labels_all -= 1

        #  assignment (shared by cls/reg/ctr) 
        # FCOS rules: inside-box  ∧  inside center window (radius = cfg*r*stride)  ∧  max(l,t,r,b) in reg_range[level]
        for n_idx, tgt in enumerate(targets):
            gt_boxes  = tgt["boxes"].to(device)   # [M,4] (xyxy)
            gt_labels = tgt["labels"].to(device)  # [M]
            if gt_boxes.numel() == 0:
                continue

            # per-gt center + area
            cx = 0.5 * (gt_boxes[:, 0] + gt_boxes[:, 2])  # [M]
            cy = 0.5 * (gt_boxes[:, 1] + gt_boxes[:, 3])  # [M]
            gt_area = ((gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) *
                    (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0))  # [M]

            for l_idx in range(L):
                HW = num_loc[l_idx]
                off = offsets[l_idx]
                pts = pts_per_level[l_idx]  # [HW,2] (x,y)
                stride_l = strides[l_idx]
                stride_l = int(stride_l.item()) if torch.is_tensor(stride_l) else int(stride_l)

                y = pts[:, 0:1]  # [HW,1]
                x = pts[:, 1:2]  # [HW,1]

                # distances to sides [HW, M]
                ldist = x - gt_boxes[:, 0].view(1, -1)
                tdist = y - gt_boxes[:, 1].view(1, -1)
                rdist = gt_boxes[:, 2].view(1, -1) - x
                bdist = gt_boxes[:, 3].view(1, -1) - y

                inside = (ldist > 0) & (tdist > 0) & (rdist > 0) & (bdist > 0)

                rad = float(self.center_sampling_radius) * float(stride_l)
                cx_ok = (x > (cx.view(1, -1) - rad)) & (x < (cx.view(1, -1) + rad))
                cy_ok = (y > (cy.view(1, -1) - rad)) & (y < (cy.view(1, -1) + rad))
                center_ok = cx_ok & cy_ok

                max_ltrb = torch.maximum(
                    torch.maximum(ldist, tdist),
                    torch.maximum(rdist, bdist)
                )
                rmin = reg_range[l_idx, 0].to(device)
                rmax = reg_range[l_idx, 1].to(device)
                range_ok = (max_ltrb >= rmin) & (max_ltrb <= rmax)

                valid = inside & center_ok & range_ok  # [HW, M]
                if not valid.any():
                    continue

                # tie-break: choose gt with smallest area per point
                area_mat = gt_area.view(1, -1).expand_as(valid).clone()
                area_mat[~valid] = float("inf")
                chosen_gt = torch.argmin(area_mat, dim=1)  # [HW]
                pos_mask_lvl = torch.isfinite(area_mat[torch.arange(HW, device=device), chosen_gt])

                if pos_mask_lvl.any():
                    idx_hw     = torch.arange(HW, device=device)[pos_mask_lvl]  # [P_lvl]
                    idx_global = off + idx_hw                                   # into [0..S)
                    chosen     = chosen_gt[pos_mask_lvl]                        # [P_lvl]

                    # pull distances for chosen gt at each positive point
                    row  = torch.arange(chosen.numel(), device=device)
                    lpos = ldist[pos_mask_lvl, :][row, chosen]
                    tpos = tdist[pos_mask_lvl, :][row, chosen]
                    rpos = rdist[pos_mask_lvl, :][row, chosen]
                    bpos = bdist[pos_mask_lvl, :][row, chosen]

                    labels_all[n_idx, idx_global] = gt_labels[chosen]  # 1..C

                    reg_tgts_all[n_idx, idx_global, :] = (
                        torch.stack([lpos, tpos, rpos, bpos], dim=1) / float(stride_l)
                    )

                    # centerness target
                    lr_min, lr_max = torch.minimum(lpos, rpos), torch.maximum(lpos, rpos)
                    tb_min, tb_max = torch.minimum(tpos, bpos), torch.maximum(tpos, bpos)
                    ctr = torch.sqrt((lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))).clamp(0.0, 1.0)
                    ctr_tgts_all[n_idx, idx_global] = ctr

        # flatten model outputs
        cls_all = _flatten_levels(cls_logits)              # [N, S, C]
        reg_all = _flatten_levels(reg_outputs)             # [N, S, 4]
        ctr_all = _flatten_levels(ctr_logits).squeeze(-1)  # [N, S]

        # positives + normalizer
        pos = (labels_all > -1)           # [N, S]
        num_pos = pos.sum().clamp(min=1) # scalar

        if DEBUG:
            print(f"[dbg] N={N} L={L} C={C} S={S}  #pos={int(num_pos)}")
            print(f"[dbg] cls_all {tuple(cls_all.shape)}  reg_all {tuple(reg_all.shape)}  ctr_all {tuple(ctr_all.shape)}")
            print(f"[dbg] labels {tuple(labels_all.shape)}  reg_tgts {tuple(reg_tgts_all.shape)}  ctr_tgts {tuple(ctr_tgts_all.shape)}")


        def _loss_cls(cls_logits_flat, labels_int, num_pos_scalar):
            """
            implement classification focal loss here.
            inputs:
            - cls_logits_flat: [N, S, C] float logits
            - labels_int:      [N, S]     long (0=bg, 1..C)
            - num_pos_scalar:  scalar tensor (#positives, min=1)
            return:
            - cls_loss: scalar tensor
            """
            print(labels_int)
            print(labels_int.max())
            #align with 0-index

            labels_0based = labels_int
            valid = labels_int > -1

            #create mapping of targets so that we can compare to cls_logits_flat
            target = torch.zeros_like(cls_logits_flat)

            #directly create one hot vector
            one_hot_vector = F.one_hot(
                labels_0based[valid],
                num_classes=cls_logits_flat.shape[2]
            ).float()
            #fix type and assign
            one_hot_vector = one_hot_vector.to(dtype=target.dtype, device=target.device)
            target[valid] = one_hot_vector 
            #making sure to use correct alpha/gamma parameters
            loss = sigmoid_focal_loss(cls_logits_flat, target, alpha=0.25, gamma=2, reduction="sum")
            
            return loss / num_pos_scalar


        def _loss_reg(reg_logits_flat, reg_tgts_flat, pos_mask, num_pos_scalar):
            """
            regression loss (GIoU) on positives only.
            model predicts distances [l, t, r, b] (stride-normalized).
            we convert both pred and tgt to local xyxy = [-l, -t, r, b] and run GIoU.
            """
            # pick only positive locations
            reg_pred_pos = reg_logits_flat[pos_mask]    # [P, 4]
            reg_tgt_pos  = reg_tgts_flat[pos_mask]      # [P, 4]

            # no positives -> return a proper zero scalar on the right device/dtype
            if reg_pred_pos.numel() == 0:
                return reg_logits_flat.sum() * 0.0

            # distances -> local boxes
            pred_xyxy = torch.stack(
                [-reg_pred_pos[:, 0], -reg_pred_pos[:, 1], reg_pred_pos[:, 2], reg_pred_pos[:, 3]],
                dim=1
            )
            tgt_xyxy = torch.stack(
                [-reg_tgt_pos[:, 0], -reg_tgt_pos[:, 1], reg_tgt_pos[:, 2], reg_tgt_pos[:, 3]],
                dim=1
            )

            # giou over positives, normalize by #positives
            loss = giou_loss(pred_xyxy, tgt_xyxy, reduction="sum") / num_pos_scalar
            return loss


        def _loss_ctr(ctr_logits_flat, ctr_tgts_flat, pos_mask, num_pos_scalar):
            """
            centerness loss (BCE-with-logits) on positives only.
            logits are [N, S]; targets are in (0,1).
            """
            ctr_logit_pos = ctr_logits_flat[pos_mask]   # [P]
            ctr_tgt_pos   = ctr_tgts_flat[pos_mask]     # [P]

            # no positives -> proper zero scalar
            if ctr_logit_pos.numel() == 0:
                return ctr_logits_flat.sum() * 0.0

            # bce-with-logits over positives, normalize by #positives
            loss = F.binary_cross_entropy_with_logits(
                ctr_logit_pos, ctr_tgt_pos, reduction="sum"
            ) / num_pos_scalar
            return loss

        # call the three stubs 
        cls_loss = _loss_cls(cls_all, labels_all, num_pos)
        reg_loss = _loss_reg(reg_all, reg_tgts_all, pos, num_pos)
        ctr_loss = _loss_ctr(ctr_all, ctr_tgts_all, pos, num_pos)

        final_loss = cls_loss + reg_loss + ctr_loss

        # print("TEST")
        # print(cls_loss)
        # print(reg_loss)
        # print(ctr_loss)
        
        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": final_loss,
        }







    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with low object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes and their labels
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep a fixed number of boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        detections = []
        for _ in range(len(image_shapes)):
            detections.append({"boxes": [],
                "scores": [],
                "labels": [],
            })

        # Loop over every pyramid level
        for pts, stride, cls_tensor, reg_tensor, ctr_tensor in zip(points, strides, cls_logits, reg_outputs, ctr_logits):
            
            # compute the object scores
            cls_prob = cls_tensor.sigmoid()
            centerness_prob = ctr_tensor.sigmoid()
            object_scores = (cls_prob * centerness_prob).sqrt()

            # filter out boxes with low object scores
            object_scores_mask = object_scores > self.score_thresh

            # Loopiong over every image in the batch
            for i, (scores, score_mask, reg_tensor_, image_shape) in enumerate(zip(object_scores, object_scores_mask, reg_tensor, image_shapes)):
                filtered_scores = scores[score_mask]

                if filtered_scores.numel() == 0:
                    continue

                # select the top K boxes
                if len(filtered_scores) < self.topk_candidates:
                    topk = len(filtered_scores)
                    topk_box_scores, topk_idxs = torch.topk(filtered_scores, topk)

                else:
                    topk_box_scores, topk_idxs = torch.topk(filtered_scores, self.topk_candidates)

                # All coordinates (indices) where x >= threshold
                all_coords = score_mask.nonzero(as_tuple=False)  # shape [num_valid, 3]

                # Pick only those corresponding to top-k (shape [k, (C x H x W)])
                topk_boxes_coords = all_coords[topk_idxs]

                # get ltrb boxes
                ltrb_boxes = reg_tensor_[:, topk_boxes_coords[:, -2], topk_boxes_coords[:, -1]].t()

                # convert ltrb to (x0, y0, x1, y1) producing tensor shape of (k, 4)
                decoded_boxes = ltrb_boxes * stride  
                
                # pts is size H x W x 2 with 2d coordinates in format of (height, width)
                # x is width and y is height 
                decoded_boxes = torch.stack([
                    pts[topk_boxes_coords[:, -2], topk_boxes_coords[:, -1], 1] - decoded_boxes[:, 0],   #x0 = x - ls
                    pts[topk_boxes_coords[:, -2], topk_boxes_coords[:, -1], 0] - decoded_boxes[:, 1],   #y0 = y - ts
                    pts[topk_boxes_coords[:, -2], topk_boxes_coords[:, -1], 1] + decoded_boxes[:, 2],   #x1 = x + rs
                    pts[topk_boxes_coords[:, -2], topk_boxes_coords[:, -1], 0] + decoded_boxes[:, 3],   #y1 = y + bs
                ], dim=1)

                labels = topk_boxes_coords[:, 0] + 1  # offset by +1

                # image_shapes (height, Width)?
                decoded_boxes[:,0] = torch.clamp(decoded_boxes[:,0], 0, image_shape[1])
                decoded_boxes[:,1] = torch.clamp(decoded_boxes[:,1], 0, image_shape[0])
                decoded_boxes[:,2] = torch.clamp(decoded_boxes[:,2], 0, image_shape[1])
                decoded_boxes[:,3] = torch.clamp(decoded_boxes[:,3], 0, image_shape[0])

                widths = decoded_boxes[:,2] - decoded_boxes[:,0]
                heights = decoded_boxes[:,3] - decoded_boxes[:,1]

                box_size_mask = heights*widths >4  # Area of boxs is greater than 4

                # box_size_mask = torch.logical_and((widths <= image_shape[0]), (widths >1))      # x1 - x0 (width) <= image width
                # box_size_mask = torch.logical_and(box_size_mask, (heights <= image_shape[1]))   # y1 - y0 (height) <= image height
                # box_size_mask = torch.logical_and(box_size_mask, (heights >1))                  # width > 1
          
                decoded_boxes = decoded_boxes[box_size_mask]
                labels = labels[box_size_mask]
                final_scores = topk_box_scores[box_size_mask]

                detections[i]["boxes"].extend(decoded_boxes)
                detections[i]["labels"].extend(labels)
                detections[i]["scores"].extend(final_scores)

        # convert list to tensor
        for i, detection in enumerate(detections):
            if len(detection["boxes"]) == 0:
                # create empty tensors on the right device
                device = points[0].device
                detections[i]["boxes"]  = torch.zeros((0, 4), device=device, dtype=torch.float32)
                detections[i]["labels"] = torch.zeros((0,),    device=device, dtype=torch.long)
                detections[i]["scores"] = torch.zeros((0,),    device=device, dtype=torch.float32)
                continue
            detections[i]["boxes"] = torch.stack(detection["boxes"])
            detections[i]["labels"] = torch.stack(detection["labels"])
            detections[i]["scores"] = torch.stack(detection["scores"])

            nms_filtered_mask = batched_nms(detections[i]["boxes"], detections[i]["scores"], detections[i]["labels"], self.nms_thresh)

            detections[i]["boxes"] = detections[i]["boxes"][nms_filtered_mask]
            detections[i]["labels"] = detections[i]["labels"][nms_filtered_mask]
            detections[i]["scores"] = detections[i]["scores"][nms_filtered_mask]

            # keep a fixed number of boxes after NMS
            if len(detections[i]["boxes"]) > self.detections_per_img:
                detections[i]["boxes"] = detections[i]["boxes"][:self.detections_per_img]
                detections[i]["labels"] = detections[i]["labels"][:self.detections_per_img]
                detections[i]["scores"] = detections[i]["scores"][:self.detections_per_img]
                

        return detections