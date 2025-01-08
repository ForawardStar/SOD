import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from models.utils import ConvBNReLU, ReceptiveConv, mean_pooling
from models.vgg import vgg16
from models.resnet import resnet50, resnet101, resnet152, Bottleneck
from models.MobileNetV2 import mobilenetv2
try:
    from models.p2t import p2t_tiny, p2t_small
except:
    print(" code is not loaded, please check the installation of PyTorch>=1.7, timm>=0.3.2!")

class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(MyConv2D, self).__init__()
        self.weight = torch.ones((out_channels, in_channels, kernel_size, kernel_size)).cuda()
        self.bias = torch.zeros(out_channels).cuda()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')

        return s.format(**self.__dict__)

class MetaNet(nn.Module):
    def __init__(self, param_dict):
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)
        self.hidden = nn.Linear(128, 64*self.param_num)
        self.hidden.weight.data.fill_(1)
        self.hidden.bias.data.fill_(0)

        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            fc_noshare = nn.Linear(64, params)
            fc_noshare.weight.data.fill_(1)
            fc_noshare.bias.data.fill_(0)
            setattr(self, 'fc{}'.format(i+1), fc_noshare)


    def forward(self, mean_features):
        hidden = F.relu(self.hidden(mean_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i+1))
            filters[name] = fc(hidden[:, i * 64:(i + 1) * 64])
        return filters

class AdaptiveFeatureAugmentation(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveFeatureAugmentation, self).__init__()
        hidden_dim = in_channels
        self.AttributionPredictor = nn.Sequential(ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), nn.Sigmoid())

        self.ThresholdPredictor = nn.Sequential(ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), ConvBNReLU(hidden_dim, hidden_dim, stride=1, pad=1, groups=hidden_dim), nn.Sigmoid())

        self.processor = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1)

        self.alpha = 30.0

    def forward(self, x):
        attributions = self.AttributionPredictor(x)
        thresholds = self.ThresholdPredictor(x)

        return 0.5 * attributions * (1 + F.tanh(self.alpha * (attributions - thresholds)))

class AdaptiveCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveCrossAttention, self).__init__()
        hidden_dim = in_channels
        self.conv11 = nn.Conv2d(in_channels*2, in_channels, 1)
        self.proj_q = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 2, 4, stride = 4), MyConv2D(hidden_dim // 2, hidden_dim // 2, 2, stride=2), nn.Conv2d(hidden_dim // 2, hidden_dim, 2, stride = 2))
        self.proj_k = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 2, 4, stride = 4), MyConv2D(hidden_dim // 2, hidden_dim // 2, 2, stride=2), nn.Conv2d(hidden_dim // 2, hidden_dim, 2, stride = 2))
        self.proj_v = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 2, 4, stride = 4), MyConv2D(hidden_dim // 2, hidden_dim // 2, 2, stride=2), nn.Conv2d(hidden_dim // 2, hidden_dim, 2, stride = 2))

        self.mlp = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU())

    def get_param_dict(self):
        """Find all the MyConv2D layers, and compute the number of parameter they need"""
        param_dict = defaultdict(int)

        def dfs(module, name):
            for name2, layer in module.named_children():
                dfs(layer, '%s.%s' % (name, name2) if name != '' else name2)
            if module.__class__ == MyConv2D:
                param_dict[name] += int(np.prod(module.weight.shape))
                param_dict[name] += int(np.prod(module.bias.shape))

        dfs(self, '')
        return param_dict

    def forward(self, F_b, F_l):
        batch, d_b, h_b, w_b = F_b.shape[0], F_b.shape[1], F_b.shape[2], F_b.shape[3]
        d_l, h_l, w_l = F_l.shape[1], F_l.shape[2], F_l.shape[3]

        pre_Q = self.proj_q(F.layer_norm(F_l, normalized_shape=[w_l]))
        pre_K = self.proj_k(F.layer_norm(F_b, normalized_shape=[w_b]))
        pre_V = self.proj_v(F.layer_norm(F_b, normalized_shape=[w_b]))

        Q = pre_Q.view(batch, d_l, -1).permute(0, 2, 1)
        K = pre_K.view(batch, d_b, -1)
        V = pre_V.view(batch, d_b, -1).permute(0, 2, 1)

        attention = F.softmax(Q @ K, dim=1) @ V
        pre_attention = attention.permute(0, 2, 1).view(batch, d_l, pre_Q.shape[2], pre_Q.shape[3])
        F_att = self.conv11(torch.cat([F.interpolate(F_l, size=(h_l, w_l), mode='bilinear'), F.interpolate(F_b, size=(h_l, w_l), mode='bilinear')], dim=1)) + F.interpolate(pre_attention, size=(h_l, w_l), mode='bilinear')
        
        return self.mlp(F.layer_norm(F_att, normalized_shape=[w_l]))

    def set_my_attr(self, name, value):
        # 下面这个循环是一步步遍历类似 residuals.0.conv.1 的字符串，找到相应的权值
        target = self
        for x in name.split('.'):
            if x.isnumeric():
                target = target.__getitem__(int(x))
            else:
                target = getattr(target, x)

        # 设置对应的权值
        n_weight = np.prod(target.weight.shape)
        target.weight = value[:n_weight].view(target.weight.shape)
        target.bias = value[n_weight:].view(target.bias.shape)

    def set_weights(self, weights, i=0):
        """输入权值字典，对该网络所有的 MyConv2D 层设置权值"""
        
        for name, param in weights.items():
            self.set_my_attr(name, weights[name][i])


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SOD(nn.Module):
    def __init__(self, arch='mobilenetv2', pretrained=True, use_carafe=True,
                 enc_channels=[64, 128, 256, 512, 512, 256, 256],
                 dec_channels=[32, 64, 128, 128, 256, 256, 256], freeze_s1=False):
        super(SOD, self).__init__()
        
        self.arch = arch
        self.backbone_mobile = eval("mobilenetv2")(pretrained)
        self.backbone_p2t = eval("p2t_small")(pretrained)

        self.conv11_mobile = nn.Conv2d(168, 64, 1)
        self.conv11_p2t = nn.Conv2d(1024, 64, 1)

        self.afa_mobile = AdaptiveFeatureAugmentation(64)
        self.afa_p2t = AdaptiveFeatureAugmentation(64)

        self.aca = AdaptiveCrossAttention(64)
        self.metanet = MetaNet(self.aca.get_param_dict())



        if arch == 'vgg16':
            enc_channels=[64, 128, 256, 512, 512, 256, 256]#, 256, 256]
        elif 'resnet50' in arch:
            enc_channels=[64, 256, 512, 1024, 2048, 1024, 1024]
            dec_channels=[32, 64, 256, 512, 512, 128, 128]
        elif 'mobilenetv2' in arch:
            enc_channels=[16, 24, 32, 96, 160, 40, 40]
            dec_channels=[16, 24, 32, 40, 40, 40, 40]
        elif 'p2t_small' in arch:
            enc_channels=[64, 128, 320, 512, 256, 256]
            dec_channels=[32, 64, 128, 256, 128, 128]
        

        use_dwconv = 'mobilenet' in arch
        
       # if 'vgg' in arch or 'p2t' in arch:
        self.conv6 = nn.Sequential(nn.MaxPool2d(2,2,0),
                                    ConvBNReLU(64, 64),                                   
                                    ConvBNReLU(64, 256, residual=False),
                                    )
        self.conv7 = nn.Sequential(nn.MaxPool2d(2,2,0),
            ConvBNReLU(256, enc_channels[-1]),
                                       ConvBNReLU(enc_channels[-1], enc_channels[-1], residual=False),
                                      )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fpn = CustomDecoder(enc_channels, dec_channels, use_dwconv=use_dwconv)
        
        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(dec_channels[4], 1, 1, stride=1, padding=0)
        self._freeze_backbone(freeze_s1=freeze_s1)
        
    
    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1
        groups = 1
        expansion = 4
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=groups,
                                base_width=self.base_width, dilation=1,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _freeze_backbone(self, freeze_s1):
        if not freeze_s1:
            return
        assert('resnet' in self.arch and '3x3' not in self.arch)
        m = [self.backbone.conv1, self.backbone.bn1, self.backbone.relu]
        print("freeze stage 0 of resnet")
        for p in m:
            for pp in p.parameters():
                p.requires_grad = False

    def forward(self, input):
        
        backbone_features_mobile = self.backbone_mobile(input)
        size_mobile = backbone_features_mobile[0].shape[2]
        features_concat_mobile = torch.cat([backbone_features_mobile[0], F.interpolate(backbone_features_mobile[1], size=(size_mobile, size_mobile), mode='bilinear'), F.interpolate(backbone_features_mobile[2], size=(size_mobile, size_mobile), mode='bilinear'), F.interpolate(backbone_features_mobile[3], size=(size_mobile, size_mobile), mode='bilinear')], dim=1)
       
        backbone_features_p2t = self.backbone_p2t(input)
        size_p2t = backbone_features_p2t[0].shape[2]
        features_concat_p2t = torch.cat([backbone_features_p2t[0], F.interpolate(backbone_features_p2t[1], size=(size_p2t, size_p2t), mode='bilinear'), F.interpolate(backbone_features_p2t[2], size=(size_p2t, size_p2t), mode='bilinear'), F.interpolate(backbone_features_p2t[3], size=(size_p2t, size_p2t), mode='bilinear')], dim=1)

        
        #features_concat = self.afa_mobile(self.conv11_mobile(features_concat_mobile)) + F.interpolate(self.afa_p2t(self.conv11_p2t(features_concat_p2t)), size=(size_mobile, size_mobile), mode='bilinear')
        features_mobile = self.conv11_mobile(features_concat_mobile)
        features_p2t = self.conv11_p2t(features_concat_p2t)
        features_mean_pooling = mean_pooling([features_mobile, features_p2t])
        params = self.metanet(features_mean_pooling)
        self.aca.set_weights(params, 0)
        features_concat = self.aca(F.interpolate(self.afa_mobile(features_mobile), size=(int(size_p2t / 2), int(size_p2t / 2))), F.interpolate(self.afa_p2t(features_p2t), size=(int(size_p2t / 2), int(size_p2t / 2))))
        ed1 = self.conv6(features_concat)
        ed2 = self.conv7(ed1)
        attention = torch.sigmoid(self.gap(ed2))

        features = self.fpn(backbone_features_p2t + [ed1, ed2], attention)
        
        saliency_maps = []
        for idx, feature in enumerate(features[:5]):
            saliency_maps.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1))(feature),
                    input.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )
            # p2t can alternatively use features of 4 levels. Here 5 levels are applied.

        return torch.sigmoid(torch.cat(saliency_maps, dim=1))


class CustomDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_dwconv=False):
        super(CustomDecoder, self).__init__()
        self.inners_a = nn.ModuleList()
        self.inners_b = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0))
        
        self.fuse = nn.ModuleList()
        dilation = [[1, 2, 4, 8]] * (len(in_channels) - 4) + [[1, 2, 3, 4]] * 2 + [[1, 1, 1, 1]] * 2
        baseWidth = [32] * (len(in_channels) - 5) + [24] * 5
        print("using dwconv:", use_dwconv)
        for i in range(len(in_channels)):
            self.fuse.append(nn.Sequential(
                    ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=use_dwconv),
                    ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=use_dwconv)))

    def forward(self, features, att=None):
        if att is not None:
            stage_result = self.fuse[-1](self.inners_a[-1](features[-1] * att))
        else:
            stage_result = self.fuse[-1](self.inners_a[-1](features[-1]))
        results = [stage_result]
        num_mul_att = 1
        for idx in range(len(features) - 2, -1, -1):
            inner_top_down = F.interpolate(self.inners_b[idx](stage_result),
                                           size=features[idx].shape[2:],
                                           mode='bilinear',
                                           align_corners=False)
            if att is not None and att.shape[1] == features[idx].shape[1] and num_mul_att:
                features[idx] = features[idx] * att
                num_mul_att -= 1
            inner_lateral = self.inners_a[idx](features[idx])
            stage_result = self.fuse[idx](torch.cat((inner_top_down, inner_lateral), dim=1))
            results.insert(0, stage_result)

        return results
