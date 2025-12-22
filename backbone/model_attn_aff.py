import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer
import numpy as np

from backbone.wavemlp import PATM,WaveBlock


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WeTr2(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)
        self.classifier2 = nn.Conv2d(self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
        #               bias=False),
        #     #PATM(self.num_classes - 1, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # )
        #Mlp(512)
        self.wave = PATM(self.num_classes - 1, qkv_bias=False, qk_scale=None, attn_drop=0.,mode='fc')
        #self.wave = WaveBlock(512, mlp_ratio=4., qkv_bias=False, qk_scale=None,
         #         attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, mode='fc')
    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # for param in list(self.classifier.parameters()):
        #     param_groups[2].append(param)
        for param in list(self.wave.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)
        param_groups[2].append(self.classifier2.weight)



        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    #2
    def forward(self, x, cam_only=False, seg_detach=True, affine = False,  ):

        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x

        seg = self.decoder(_x)
        # seg = self.decoder(_x4)
        # print('--------')

        attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        attn_pred = self.attn_proj(attn_cat)
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]

        # _x4 = self.dropout(_x4.clone()
        cls_x4 = self.pooling(_x4, (1, 1))
        #cls_x4 = cls_x4.clone()
        #.detach()#.clone()#.detach()
        cls_x4 = self.classifier(cls_x4)
        wave = self.wave(cls_x4)
        cls_x4 = self.classifier2(wave)
        if cam_only:
            cam_s4 = F.conv2d(wave, self.classifier2.weight).detach()
            return cam_s4, attn_pred

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        if affine:
            return cls_x4, seg, _attns


        return cls_x4, seg, _attns, attn_pred


class WeTr599(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)
        self.attn_proj1 = nn.Conv2d(in_channels=512, out_channels=8, kernel_size=1, bias=True)
        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        self.classifier2 = nn.Conv2d(self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
        #               bias=False),
        #     #PATM(self.num_classes - 1, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # )
        # Mlp(512)
        self.wave = PATM(512, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # self.wave = WaveBlock(512, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        #         attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, mode='fc')

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # for param in list(self.classifier.parameters()):
        #     param_groups[2].append(param)
        for param in list(self.wave.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)
        param_groups[2].append(self.attn_proj1.weight)
        param_groups[2].append(self.attn_proj1.bias)
        param_groups[2].append(self.classifier2.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    # 1
    def forward(self, x, cam_only=False, seg_detach=True, affine=False, ):
        #print(x.shape)torch.Size([2, 3, 320, 320])
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        #print('_x4', _x4.shape)_x4 torch.Size([2, 512, 20, 20])
        #_x4 = self.wave(_x4)
        #_x = torch.cat(_x1, _x2, _x3, _x4)
        seg = self.decoder(_x)
        # seg = self.decoder(_x4)
        # print('--------')
        # print('_attns[-1]', _attns[-1].shape)
        # print('_attns[-2]',_attns[-2].shape)
        # _attns[-1] torch.Size([2, 8, 400, 400])
        # _attns[-2] torch.Size([2, 8, 400, 400])
        #_x4a = _x4.reshape(_x4.shape[0], 8, 160,160)

        #print('_x4a', _x4a.shape)#_x4a torch.Size([8, 8, 400, 400])
        #attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()
        #print('attn_cat', attn_cat.shape)
        _x4a = self.attn_proj1(_x4)
        _x4a = F.interpolate(_x4a, size=(_attns[-1].shape[3], _attns[-1].shape[3]), mode='bilinear',
                            align_corners=True)
        attn_cat = torch.cat((_attns[-1],_x4a), dim=1)
        attn_pred = self.attn_proj(attn_cat)
        #print( attn_pred.shape)#[2, 1, 400, 400])
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]


        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred


        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        if affine:
            return cls_x4, seg, _attns

        return cls_x4, seg, _attns, attn_pred


class WeTr522(nn.Module):
    def forward(self, x, cam_only=False, seg_detach=True, affine=False, ):
        #print(x.shape)torch.Size([2, 3, 320, 320])
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        #print('_x4', _x4.shape)_x4 torch.Size([2, 512, 20, 20])
        #_x4 = self.wave(_x4)
        #_x = torch.cat(_x1, _x2, _x3, _x4)
        seg = self.decoder(_x)
        _x4a = self.attn_proj1(_x4)
        _x4a = F.interpolate(_x4a, size=(_attns[-1].shape[3], _attns[-1].shape[3]), mode='bilinear',
                            align_corners=True)
        attn_pred = self.attn_proj(_x4a)
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        if affine:
            return cls_x4, seg, _attns
        return cls_x4, seg, _attns, attn_pred


class WeTr603(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)
        self.attn_proj1 = nn.Conv2d(in_channels=512, out_channels=8, kernel_size=1, bias=True)
        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        self.classifier2 = nn.Conv2d(self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)

        self.wave = PATM(512, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # self.wave = WaveBlock(512, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        #         attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, mode='fc')

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # for param in list(self.classifier.parameters()):
        #     param_groups[2].append(param)
        for param in list(self.wave.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)
        param_groups[2].append(self.attn_proj1.weight)
        param_groups[2].append(self.attn_proj1.bias)
        param_groups[2].append(self.classifier2.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    # 1
    def forward(self, x, cam_only=False, seg_detach=True, affine=False, ):
        #print(x.shape)torch.Size([2, 3, 320, 320])
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        #print('_x4', _x4.shape)_x4 torch.Size([2, 512, 20, 20])
        #_x4 = self.wave(_x4)
        #_x = torch.cat(_x1, _x2, _x3, _x4)
        seg = self.decoder(_x)

        _x4a = self.attn_proj1(_x4)
        _x4a = F.interpolate(_x4a, size=(_attns[-1].shape[3], _attns[-1].shape[3]), mode='bilinear',
                            align_corners=True)
        attn_cat = torch.cat((_attns[-2],_x4a), dim=1)
        attn_pred = self.attn_proj(attn_cat)
        #print( attn_pred.shape)#[2, 1, 400, 400])
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        if affine:
            return cls_x4, seg, _attns

        return cls_x4, seg, _attns, attn_pred



class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)
        self.attn_proj1 = nn.Conv2d(in_channels=512, out_channels=8, kernel_size=1, bias=True)
        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        self.classifier2 = nn.Conv2d(self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)

        self.wave = PATM(512, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        # self.wave = WaveBlock(512, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        #         attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, mode='fc')

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # for param in list(self.classifier.parameters()):
        #     param_groups[2].append(param)
        for param in list(self.wave.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)
        param_groups[2].append(self.attn_proj1.weight)
        param_groups[2].append(self.attn_proj1.bias)
        param_groups[2].append(self.classifier2.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    # 1
    def forward(self, x, cam_only=False, seg_detach=True, affine=False, ):
        #print(x.shape)torch.Size([2, 3, 320, 320])
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        #print('_x4', _x4.shape)_x4 torch.Size([2, 512, 20, 20])
        _x4 = self.wave(_x4)
        #_x = torch.cat(_x1, _x2, _x3, _x4)
        seg = self.decoder(_x)

        _x4a = self.attn_proj1(_x4)
        _x4a = F.interpolate(_x4a, size=(_attns[-1].shape[3], _attns[-1].shape[3]), mode='bilinear',
                            align_corners=True)
        attn_cat = torch.cat((_attns[-2],_x4a), dim=1)
        attn_pred = self.attn_proj(attn_cat)
        #print( attn_pred.shape)#[2, 1, 400, 400])
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        if affine:
            return cls_x4, seg, _attns

        return cls_x4, seg, _attns, attn_pred
