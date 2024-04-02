import torch.nn as nn
import torch
import math
import os
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.cuda.amp import autocast
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
import torch.nn.functional as F
from retinanet import model

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        # ex: x.shape=[4, 256, 80, 116]
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)
        # out.shape=[4, 9*80, 80, 116]
        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        # out1.shape=[4, 80, 116, 720]
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape
        # out2.shape=[4, 80, 116, 9, 80]
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        # out2.shape=[4, 80*116*9, 80]
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class GCBlock(nn.Module):
    def __init__(self, teacher_channels=256):
        super(GCBlock, self).__init__()

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))

        # self.reset_parameters()

    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context
    
    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        return rela_loss
    
    def reset_parameters(self):
        # kaiming_init(self.conv_mask_s, mode='fan_in')
        # kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)

    def forward(self, preds_S, preds_T):
        return self.get_rela_loss(preds_S, preds_T)
    
class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, teacher_path=r'.\results\teacher_model\coco_retinanet_10.pt'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.GCBlockModel = GCBlock()
        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        prior = 0.01

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # He_kiming_init
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.zero_()

        self.freeze_bn()

        try:
            self.teacher_model = model.resnet101(num_classes=num_classes, pretrained=False)
            self.teacher_model.load_state_dict(torch.load(teacher_path).state_dict())
            if torch.cuda.is_available():
                self.teacher_model = self.teacher_model.cuda()
            self.teacher_model.training = False
            self.teacher_model.eval()
            self.teacher_model.freeze_bn()
        except FileNotFoundError:
            raise FileExistsError(f'Invalid Teacher Model Path: {teacher_path}')
        except TypeError:
            raise TypeError(f'Invalid Teacher model structure.')
        except Exception as E:
            raise ValueError(E)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_teacher_feature(self, img):
        with torch.no_grad():
            x = self.teacher_model.conv1(img)
            x = self.teacher_model.bn1(x)
            x = self.teacher_model.relu(x)
            x = self.teacher_model.maxpool(x)

            x1 = self.teacher_model.layer1(x)
            x2 = self.teacher_model.layer2(x1)
            x3 = self.teacher_model.layer3(x2)
            x4 = self.teacher_model.layer4(x3)
            features = self.teacher_model.fpn([x2, x3, x4])
        return features
    
    def FGD_loss(self, t_feature, s_feature, annotation):
        # get_attention return Spatial attention and channel attention
        t_S, t_C = self.get_attention(t_feature, 0.5)
        s_S, s_C = self.get_attention(s_feature, 0.5)

        # attention loss
        attention_loss = torch.sum(torch.abs((s_C-t_C)))/len(s_C) + torch.sum(torch.abs((s_S-t_S)))/len(s_S)

        # feature loss
        Mask_fg = torch.zeros_like(t_S)
        Mask_bg = torch.ones_like(t_S)
        Mask_fg, Mask_bg = self.get_feature_loss_mask(Mask_fg, Mask_bg, annotation, t_feature.shape)

        feature_loss = self.get_feature_loss(s_feature, t_feature, Mask_fg, Mask_bg, t_C, t_S)

        # global loss
        global_loss = self.GCBlockModel(t_feature, s_feature)

        return global_loss, attention_loss, feature_loss

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention

    def get_feature_loss_mask(self, Mask_fg, Mask_bg, gt_bboxes, img_metas):
        # feature map size = original image size/8
        N, _, H, W = img_metas
        img_W = W*8
        img_H = H*8
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_W*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_W*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_H*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_H*H

            wmin = torch.floor(new_boxxes[:, 0]).int()
            wmax = torch.ceil(new_boxxes[:, 2]).int()
            hmin = torch.floor(new_boxxes[:, 1]).int()
            hmax = torch.ceil(new_boxxes[:, 3]).int()

            area = 1.0/(hmax.view(1,-1)+1-hmin.view(1,-1))/(wmax.view(1,-1)+1-wmin.view(1,-1))
            for j in range(len(gt_bboxes[i])):
                h_slice = slice(hmin[j], hmax[j]+1)
                w_slice = slice(wmin[j], wmax[j]+1)
                
                # 使用in-place操作替代torch.maximum()
                Mask_fg[i][h_slice, w_slice] = \
                        torch.maximum(Mask_fg[i][hmin[j]:hmax[j]+1, wmin[j]:wmax[j]+1], area[0][j])

            Mask_bg[i][Mask_fg[i] > 0] = 0
            sum_bg = torch.sum(Mask_bg[i])

            if sum_bg.item() != 0:
                Mask_bg[i] /= sum_bg

        
        return Mask_fg, Mask_bg

    def get_feature_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, t_C, t_S):
        loss_mse = nn.MSELoss(reduction='sum')
        
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        t_C = t_C.unsqueeze(dim=-1)
        t_C = t_C.unsqueeze(dim=-1)

        t_S = t_S.unsqueeze(dim=1)

        fea_t= torch.mul(preds_T, torch.sqrt(t_S))
        fea_t = torch.mul(fea_t, torch.sqrt(t_C))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(t_S))
        fea_s = torch.mul(fea_s, torch.sqrt(t_C))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

        return fg_loss, bg_loss

    def scale_mask(self, annotation):
        pass

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        
        if self.training:

            classification_loss, regression_loss = self.focalLoss(classification, regression, anchors, annotations)

            teacher_features = self.get_teacher_feature(img_batch)
            # With distillation method, bbox reg and cls should compute distillation loss rather than original loss func.
            # However, these terms are not added in FGD implement(not sure). So I skip computing these here.
            # t_regression = torch.cat([self.teacher_model.regressionModel(feature) for feature in teacher_features], dim=1)
            # t_classification = torch.cat([self.teacher_model.classificationModel(feature) for feature in teacher_features], dim=1)
            global_losses = []
            attention_losses = []
            fg_feature_losses = []
            bg_feature_losses = []

            for t_feature, s_feature in zip(teacher_features, features):
                global_loss, attention_loss, feature_loss = self.FGD_loss(t_feature, s_feature, annotations)
                fg_feature_loss, bg_feature_loss = feature_loss
                global_losses.append(global_loss)
                attention_losses.append(attention_loss)
                fg_feature_losses.append(fg_feature_loss)
                bg_feature_losses.append(bg_feature_loss)
                
            loss1 = torch.stack(global_losses).mean(dim=0, keepdim=True)
            loss2 = torch.stack(attention_losses).mean(dim=0, keepdim=True)
            loss3 = torch.stack(fg_feature_losses).mean(dim=0, keepdim=True)
            loss4 = torch.stack(bg_feature_losses).mean(dim=0, keepdim=True)

            return classification_loss, regression_loss, loss1, loss3, loss4, loss2
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]



def FGD_resnet18(num_classes, teacher_path, pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], teacher_path, **kwargs)
    if pretrained:
        print('loading pretrain model')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def FGD_resnet34(num_classes, teacher_path, pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], teacher_path, **kwargs)
    if pretrained:
        print('loading pretrain model')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def FGD_resnet50(num_classes, teacher_path, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], teacher_path, **kwargs)
    if pretrained:
        print('loading pretrain model')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def FGD_resnet101(num_classes, teacher_path, pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], teacher_path, **kwargs)
    if pretrained:
        print('loading pretrain model')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def FGD_resnet152(num_classes, teacher_path, pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], teacher_path, **kwargs)
    if pretrained:
        print('loading pretrain model')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
