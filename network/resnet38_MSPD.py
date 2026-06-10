import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import network.resnet38d
from tool import pyutils


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.view(N * C, 1, H, W)
        gx = F.conv2d(x, self.kernel_x, padding=1).view(N, C, H, W)
        gy = F.conv2d(x, self.kernel_y, padding=1).view(N, C, H, W)
        return gx, gy


class Net(network.resnet38d.Net):
    def __init__(self):
        super().__init__()
        # ---------- Original PCM components ----------
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(4096, 21, 1, bias=False)
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192 + 3, 192, 1, bias=False)
        self.fc_proj = torch.nn.Conv2d(4096, 128, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        torch.nn.init.xavier_uniform_(self.fc_proj.weight)

        # ===================== MSPF =====================
        # conv3=256, conv4=512, conv5=1024, conv6=4096
        self.mspf_proj3 = nn.Conv2d(256, 128, 1, bias=False)
        self.mspf_proj4 = nn.Conv2d(512, 128, 1, bias=False)
        self.mspf_proj5 = nn.Conv2d(1024, 128, 1, bias=False)
        self.mspf_proj6 = nn.Conv2d(4096, 128, 1, bias=False)

        # Auxiliary classifiers w^(l)_c for each level
        self.mspf_cls3 = nn.Conv2d(128, 21, 1, bias=False)
        self.mspf_cls4 = nn.Conv2d(128, 21, 1, bias=False)
        self.mspf_cls5 = nn.Conv2d(128, 21, 1, bias=False)
        self.mspf_cls6 = nn.Conv2d(128, 21, 1, bias=False)

        # Learnable fusion weights α_l
        self.mspf_fusion_logits = nn.Parameter(torch.zeros(4))

        torch.nn.init.kaiming_normal_(self.mspf_proj3.weight)
        torch.nn.init.kaiming_normal_(self.mspf_proj4.weight)
        torch.nn.init.kaiming_normal_(self.mspf_proj5.weight)
        torch.nn.init.kaiming_normal_(self.mspf_proj6.weight)
        torch.nn.init.xavier_uniform_(self.mspf_cls3.weight)
        torch.nn.init.xavier_uniform_(self.mspf_cls4.weight)
        torch.nn.init.xavier_uniform_(self.mspf_cls5.weight)
        torch.nn.init.xavier_uniform_(self.mspf_cls6.weight)

        # ===================== SAPD =====================
        # Appearance branch
        self.sapd_appearance_proj = nn.Conv2d(4096, 128, 1, bias=False)
        torch.nn.init.kaiming_normal_(self.sapd_appearance_proj.weight)

        # Structure branch (from conv3 which has 256 channels)
        self.sapd_structure_conv = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.sapd_structure_fuse = nn.Conv2d(128 * 3, 128, 1, bias=False)
        self.sapd_structure_proj = nn.Conv2d(128, 128, 1, bias=False)
        torch.nn.init.kaiming_normal_(self.sapd_structure_conv.weight)
        torch.nn.init.kaiming_normal_(self.sapd_structure_fuse.weight)
        torch.nn.init.kaiming_normal_(self.sapd_structure_proj.weight)

        self.sobel = Sobel()

        self.from_scratch_layers = [
            self.f8_3, self.f8_4, self.f9, self.fc8, self.fc_proj,
            self.mspf_proj3, self.mspf_proj4, self.mspf_proj5, self.mspf_proj6,
            self.mspf_cls3, self.mspf_cls4, self.mspf_cls5, self.mspf_cls6,
            self.sapd_appearance_proj,
            self.sapd_structure_conv, self.sapd_structure_fuse, self.sapd_structure_proj,
        ]
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):
        N, C, H, W = x.size()
        d = super().forward_as_dict(x)

        fea = self.dropout7(d['conv6'])
        cam = self.fc8(fea)
        f_proj = F.relu(self.fc_proj(fea), inplace=True)

        n, c, h, w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0

        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)

        cam_rv_down = self.PCM(cam_d_norm, f)
        cam_rv = F.interpolate(cam_rv_down, (H, W), mode='bilinear', align_corners=True)
        cam_up = F.interpolate(cam, (H, W), mode='bilinear', align_corners=True)

        # ===================== MSPF =====================
        v3 = F.normalize(F.relu(self.mspf_proj3(d['conv3'])), p=2, dim=1)
        v4 = F.normalize(F.relu(self.mspf_proj4(d['conv4'])), p=2, dim=1)
        v5 = F.normalize(F.relu(self.mspf_proj5(d['conv5'])), p=2, dim=1)
        v6 = F.normalize(F.relu(self.mspf_proj6(fea)), p=2, dim=1)

        aux_cam3 = self.mspf_cls3(v3)
        aux_cam4 = self.mspf_cls4(v4)
        aux_cam5 = self.mspf_cls5(v5)
        aux_cam6 = self.mspf_cls6(v6)

        # ===================== SAPD: Appearance =====================
        v_app = F.normalize(F.relu(self.sapd_appearance_proj(fea)), p=2, dim=1)

        # ===================== SAPD: Structure =====================
        f_str_mid = d['conv3']
        f_str = F.relu(self.sapd_structure_conv(f_str_mid))
        gx, gy = self.sobel(f_str)
        f_str_cat = torch.cat([f_str, gx, gy], dim=1)
        f_str_fused = F.relu(self.sapd_structure_fuse(f_str_cat))
        v_str = F.normalize(self.sapd_structure_proj(f_str_fused), p=2, dim=1)

        gx_s, gy_s = self.sobel(f_str_fused)
        struct_conf = torch.sqrt(gx_s ** 2 + gy_s ** 2 + 1e-8).mean(dim=1, keepdim=True)

        outputs = {
            'cam': cam_up,
            'cam_rv': cam_rv,
            'cam_rv_down': cam_rv_down,
            'f_proj': f_proj,
            'v3': v3, 'v4': v4, 'v5': v5, 'v6': v6,
            'aux_cam3': aux_cam3, 'aux_cam4': aux_cam4,
            'aux_cam5': aux_cam5, 'aux_cam6': aux_cam6,
            'v_app': v_app,
            'v_str': v_str,
            'struct_conf': struct_conf,
        }
        return outputs

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)
        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)
        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):
                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)
                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups
