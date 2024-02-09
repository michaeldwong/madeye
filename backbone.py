# Author: Zylo117

import torch
from torch import nn
import torch.nn.functional as F

from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import Anchors


class DiffEfficientNetCounter(nn.Module):
    # Weights path is for EfficientNetCounter that will be used for both current and prev CNNs
    def __init__(self,  compound_coef=0, load_weights=False, **kwargs):
        super(DiffEfficientNetCounter, self).__init__()

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

        self.backbone_net = EfficientNetCounter(self.backbone_compound_coef[compound_coef], load_weights)
        self.prev_net = EfficientNetCounter(self.backbone_compound_coef[compound_coef], load_weights)



        self.avg_pool = nn.AdaptiveMaxPool2d((2, 2))
        # Add 2 for the prev counts
        self.fc1 = nn.Linear(3776 + 2, 2)
        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, prev_image, current_image, prev_car_count, prev_person_count):

        _, pp3, pp4, pp5 = self.prev_net.backbone_net(prev_image)
        pp3_pooled = self.avg_pool(F.relu(pp3))
        pp4_pooled = self.avg_pool(F.relu(pp4))
        pp5_pooled = self.avg_pool(F.relu(pp5))
        prev_feature_vec = torch.cat((pp3_pooled.flatten(), pp4_pooled.flatten(), pp5_pooled.flatten()))

        _, cp3, cp4, cp5 = self.current_net.backbone_net(current_image)
        cp3_pooled = self.avg_pool(F.relu(cp3))
        cp4_pooled = self.avg_pool(F.relu(cp4))
        cp5_pooled = self.avg_pool(F.relu(cp5))
        current_feature_vec = torch.cat((cp3_pooled.flatten(), cp4_pooled.flatten(), cp5_pooled.flatten()))
        feature_vec = torch.cat((prev_feature_vec, current_feature_vec))


        t = torch.tensor([prev_count, prev_car_count])
        x = torch.cat((feature_vec, t) )
        x = self.leakyReLU(self.fc1(x))
        return x


class EfficientNetCounter(nn.Module):
    def __init__(self, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientNetCounter, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])
        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)
        self.avg_pool = nn.AdaptiveMaxPool2d((2, 2))
#        self.fc1 = nn.Linear(1888, 100)
        self.fc1 = nn.Linear(1280, 2)
        # One output for cars, other for people
        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        _, p3, p4, p5 = self.backbone_net(inputs)
        features = (p3, p4, p5)



        b0,b1,b2,b3,b4 = self.bifpn(features)

        b0_pooled = self.avg_pool(F.relu(b0))
        b1_pooled = self.avg_pool(F.relu(b1))
        b2_pooled = self.avg_pool(F.relu(b2))
        b3_pooled = self.avg_pool(F.relu(b3))
        b4_pooled = self.avg_pool(F.relu(b4))

        feature_vec = torch.cat((b0_pooled.flatten(), b1_pooled.flatten(), b2_pooled.flatten(), b3_pooled.flatten(), b4_pooled.flatten()))

#        p3_pooled = self.avg_pool(F.relu(p3))
#        p4_pooled = self.avg_pool(F.relu(p4))
#        p5_pooled = self.avg_pool(F.relu(p5))
#        feature_vec = torch.cat((p3_pooled.flatten(), p4_pooled.flatten(), p5_pooled.flatten()))
#
        x = self.leakyReLU(self.fc1(feature_vec))
        return x

class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)

        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

    def save_backbone_weights(self, outfile):
        torch.save(self.backbone_net.state_dict(), outfile)
