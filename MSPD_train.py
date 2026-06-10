import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F


# ===================== NPCM: Mini-Batch K-Means =====================
class MiniBatchKMeans:
    def __init__(self, K, D, momentum=0.9):
        self.K = K
        self.D = D
        self.momentum = momentum
        self.centers = F.normalize(torch.randn(K, D).cuda(), p=2, dim=1)
        self.counts = torch.zeros(K).cuda()

    @torch.no_grad()
    def update_and_assign(self, prototypes):
        sim = torch.matmul(prototypes, self.centers.t())
        assignments = sim.argmax(dim=1)
        for k in range(self.K):
            mask = assignments == k
            n = mask.sum()
            if n > 0:
                cluster_mean = prototypes[mask].mean(dim=0)
                self.counts[k] += n.float()
                self.centers[k] = F.normalize(
                    self.momentum * self.centers[k] + (1 - self.momentum) * cluster_mean,
                    p=2, dim=0
                )
            elif self.counts[k] == 0:
                self.centers[k] = F.normalize(
                    self.centers[k] + torch.randn_like(self.centers[k]) * 0.01,
                    p=2, dim=0
                )
        return assignments


# ===================== Helper functions =====================
def adaptive_min_pooling_loss(x):
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)
    return loss


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
    return x


def compute_prototypes(v_feat, cam, label, bg_threshold=0.20, topk_ratio=8):
    """
    v_feat: (N, 128, H, W) projected features (L2 normalized)
    cam: (N, 21, H, W) raw CAM
    label: (N, 21, 1, 1) class labels
    Returns: (21, 128) prototypes (L2 normalized)
    """
    N, D, H, W = v_feat.shape
    cam_relu = F.relu(cam.detach())
    cam_max = torch.max(cam_relu.view(N, 21, -1), dim=-1)[0].view(N, 21, 1, 1) + 1e-5
    cam_norm = cam_relu / cam_max
    cam_norm[:, 0, :, :] = bg_threshold
    scores = F.softmax(cam_norm * label, dim=1)
    scores_t = scores.transpose(0, 1)

    fea_flat = v_feat.permute(0, 2, 3, 1).reshape(-1, D)
    total_pixels = N * H * W
    k = total_pixels // topk_ratio

    top_values, top_indices = torch.topk(
        cam_norm.transpose(0, 1).reshape(21, -1), k=k, dim=-1
    )

    prototypes = torch.zeros(21, D, device=v_feat.device)
    for i in range(21):
        indices = top_indices[i]
        vals = top_values[i].unsqueeze(-1)
        fea_top = fea_flat[indices]
        prototypes[i] = (vals * fea_top).sum(dim=0) / (vals.sum() + 1e-10)

    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes


def compute_appearance_prototypes(v_app, cam_rv, label, bg_threshold=0.20, topk_ratio=8):
    return compute_prototypes(v_app, cam_rv, label, bg_threshold, topk_ratio)


def compute_structure_prototypes(v_str, struct_conf, pseudo_label, C=21):
    """
    v_str: (N, 128, H, W) structure embeddings
    struct_conf: (N, 128, H, W) structure confidence maps
    pseudo_label: (N, H, W) pseudo labels from CAM
    Returns: (21, 128) structure prototypes
    """
    D = v_str.shape[1]
    fea_flat = v_str.permute(0, 2, 3, 1).reshape(-1, D)
    conf_flat = struct_conf.view(-1)
    pl_flat = pseudo_label.view(-1)

    prototypes = torch.zeros(C, D, device=v_str.device)
    for c in range(C):
        mask = pl_flat == c
        n_pixels = mask.sum()
        if n_pixels == 0:
            continue
        # Top-K by structure confidence
        k = max(int(n_pixels * 0.125), 1)
        conf_at_c = conf_flat[mask]
        topk_vals, topk_idx = torch.topk(conf_at_c, k=k, dim=0)
        fea_at_c = fea_flat[mask][topk_idx]
        prototypes[c] = (topk_vals.unsqueeze(-1) * fea_at_c).sum(dim=0) / (topk_vals.sum() + 1e-10)

    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes


def npcm_p2p_loss(feat, pseudo_label, prototypes, cluster_assignments, K=21, tau=0.1, M=5):
    """
    feat: (N*H*W, D) pixel features
    pseudo_label: (N*H*W,) pseudo labels
    prototypes: (C, D) all prototypes
    cluster_assignments: (C,) which cluster each prototype belongs to
    """
    device = feat.device
    C = prototypes.shape[0]
    loss = 0.0
    count = 0

    for c in range(C):
        mask = pseudo_label == c
        n_pixels = mask.sum()
        if n_pixels < 1:
            continue

        c_cluster = cluster_assignments[c]
        # Negatives: prototypes in same cluster except c itself
        neg_indices = (cluster_assignments == c_cluster).nonzero(as_tuple=True)[0]
        neg_indices = neg_indices[neg_indices != c]
        if len(neg_indices) == 0:
            continue

        # Select hardest M negatives (highest similarity)
        pos_proto = prototypes[c:c+1]
        neg_protos = prototypes[neg_indices]
        sim = torch.matmul(feat[mask], neg_protos.t())
        hard_neg_idx = torch.topk(sim, k=min(M, len(neg_indices)), dim=1, largest=True)[1]
        hard_negs = neg_protos[hard_neg_idx]

        pos_sim = torch.matmul(feat[mask], pos_proto.t())
        neg_sim = torch.bmm(hard_negs, feat[mask].unsqueeze(-1)).squeeze(-1)

        logits_pos = pos_sim / tau
        logits_neg = neg_sim / tau
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        target = torch.zeros(len(feat[mask]), dtype=torch.long, device=device)
        loss += F.cross_entropy(logits, target)
        count += 1

    return loss / max(count, 1)


def sapd_contrast_loss(emb, proto_src, pseudo_label, tau=0.1):
    """
    emb: (N*H*W, D) pixel embeddings from one branch
    proto_src: (C, D) prototypes from the other branch (cross-branch)
    pseudo_label: (N*H*W,) pseudo labels
    """
    C = proto_src.shape[0]
    pos = proto_src[pseudo_label]
    sim_pos = torch.sum(emb * pos, dim=1)
    logits = torch.matmul(emb, proto_src.t()) / tau
    target = pseudo_label
    loss = F.cross_entropy(logits, target)
    return loss


def affinity_loss(feat, pseudo_label):
    """
    ToCo-style affinity loss L_aff
    feat: (N, D, H, W) feature map
    pseudo_label: (N, H, W) pseudo labels (will be resized to match feat spatial size)
    """
    N, D, H, W = feat.shape
    # Resize pseudo_label to match feat spatial size
    pl = F.interpolate(pseudo_label.float().unsqueeze(1), size=(H, W), mode='nearest').long().view(N, -1)

    feat_flat = feat.view(N, D, -1)
    feat_norm = F.normalize(feat_flat, p=2, dim=1)

    sim = torch.matmul(feat_norm.transpose(1, 2), feat_norm)

    loss_pos = 0
    loss_neg = 0
    for i in range(N):
        pos_mask = pl[i].unsqueeze(0) == pl[i].unsqueeze(1)
        neg_mask = pl[i].unsqueeze(0) != pl[i].unsqueeze(1)
        pos_sim = sim[i][pos_mask]
        neg_sim = sim[i][neg_mask]
        if pos_sim.numel() > 0:
            loss_pos += (1 - pos_sim).mean()
        if neg_sim.numel() > 0:
            loss_neg += neg_sim.mean()

    return loss_pos / N + loss_neg / N


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_MSPD", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_MSPD", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--bg_threshold", default=0.20, type=float)
    # NPCM
    parser.add_argument("--npcm_K", default=10, type=int, help="Number of clusters for NPCM")
    parser.add_argument("--npcm_M", default=5, type=int, help="Hard negatives per pixel")
    parser.add_argument("--npcm_momentum", default=0.9, type=float)
    # Loss weights
    parser.add_argument("--alpha_app", default=0.1, type=float)
    parser.add_argument("--beta_str", default=0.2, type=float)
    parser.add_argument("--nce_tau", default=0.1, type=float)

    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')
    print(vars(args))

    model = getattr(importlib.import_module(args.network), 'Net')()
    tblogger = SummaryWriter(args.tblog_dir)

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(448, 768),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                                          saturation=0.3, hue=0.1),
                                                   np.asarray,
                                                   model.normalize,
                                                   imutils.RandomCrop(args.crop_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   worker_init_fn=worker_init_fn)

    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter(
        'loss', 'loss_cls', 'loss_er', 'loss_ecr',
        'loss_aff', 'loss_p2p', 'loss_app', 'loss_str'
    )

    timer = pyutils.Timer("Session started: ")

    # NPCM cluster centers
    npcm_kmeans = MiniBatchKMeans(K=args.npcm_K, D=128, momentum=args.npcm_momentum)

    for ep in range(args.max_epoches):
        for iter, pack in enumerate(train_data_loader):
            img1 = pack[1]
            img2 = F.interpolate(img1, size=(128, 128), mode='bilinear', align_corners=True)
            N, C, H, W = img1.size()
            label = pack[2]

            bg_score = torch.ones((N, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            # ============ Forward ============
            out1 = model(img1)
            cam1, cam_rv1, f_proj1, cam_rv1_down = (
                out1['cam'], out1['cam_rv'], out1['f_proj'], out1['cam_rv_down']
            )
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])

            cam1_norm = visualization.max_norm(cam1)
            cam1_norm = F.interpolate(cam1_norm, size=(128, 128), mode='bilinear', align_corners=True) * label
            cam_rv1_norm = visualization.max_norm(cam_rv1)
            cam_rv1_norm = F.interpolate(cam_rv1_norm, size=(128, 128), mode='bilinear', align_corners=True) * label

            # ---- Branch 2 ----
            out2 = model(img2)
            cam2, cam_rv2, f_proj2, cam_rv2_down = (
                out2['cam'], out2['cam_rv'], out2['f_proj'], out2['cam_rv_down']
            )
            label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label)[:, 1:, :, :])
            cam2_norm = visualization.max_norm(cam2) * label
            cam_rv2_norm = visualization.max_norm(cam_rv2) * label

            # ============ Classification loss L_cls ============
            loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :], label[:, 1:, :, :])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :], label[:, 1:, :, :])
            loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2

            # ============ Equivariance loss (SEAM-style) ============
            ns, cs, hs, ws = cam2_norm.size()
            loss_er = torch.mean(torch.abs(cam1_norm[:, 1:, :, :] - cam2_norm[:, 1:, :, :]))

            cam1_norm[:, 0, :, :] = 1 - torch.max(cam1_norm[:, 1:, :, :], dim=1)[0]
            cam2_norm[:, 0, :, :] = 1 - torch.max(cam2_norm[:, 1:, :, :], dim=1)[0]

            tensor_ecr1 = torch.abs(max_onehot(cam2_norm.detach()) - cam_rv1_norm)
            tensor_ecr2 = torch.abs(max_onehot(cam1_norm.detach()) - cam_rv2_norm)
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            # ============ MSPF: Multi-level prototypes ============
            # Downsample CAMs to match each level
            v3, v4, v5, v6 = out1['v3'], out1['v4'], out1['v5'], out1['v6']
            aux_cam3, aux_cam4, aux_cam5, aux_cam6 = (
                out1['aux_cam3'], out1['aux_cam4'], out1['aux_cam5'], out1['aux_cam6']
            )
            v3_2, v4_2, v5_2, v6_2 = out2['v3'], out2['v4'], out2['v5'], out2['v6']
            aux_cam3_2, aux_cam4_2, aux_cam5_2, aux_cam6_2 = (
                out2['aux_cam3'], out2['aux_cam4'], out2['aux_cam5'], out2['aux_cam6']
            )

            with torch.no_grad():
                protos_3 = compute_prototypes(v3, aux_cam3, label, args.bg_threshold)
                protos_4 = compute_prototypes(v4, aux_cam4, label, args.bg_threshold)
                protos_5 = compute_prototypes(v5, aux_cam5, label, args.bg_threshold)
                protos_6 = compute_prototypes(v6, aux_cam6, label, args.bg_threshold)

                protos_3_2 = compute_prototypes(v3_2, aux_cam3_2, label, args.bg_threshold)
                protos_4_2 = compute_prototypes(v4_2, aux_cam4_2, label, args.bg_threshold)
                protos_5_2 = compute_prototypes(v5_2, aux_cam5_2, label, args.bg_threshold)
                protos_6_2 = compute_prototypes(v6_2, aux_cam6_2, label, args.bg_threshold)

                # Fuse prototypes with learnable weights
                fusion_w = F.softmax(model.module.mspf_fusion_logits, dim=0)
                proto_fused = (
                    fusion_w[0] * protos_3 + fusion_w[1] * protos_4 +
                    fusion_w[2] * protos_5 + fusion_w[3] * protos_6
                )
                proto_fused = F.normalize(proto_fused, p=2, dim=1)

                proto_fused_2 = (
                    fusion_w[0] * protos_3_2 + fusion_w[1] * protos_4_2 +
                    fusion_w[2] * protos_5_2 + fusion_w[3] * protos_6_2
                )
                proto_fused_2 = F.normalize(proto_fused_2, p=2, dim=1)

            # ============ SAPD: Appearance prototypes ============
            with torch.no_grad():
                v_app1 = out1['v_app']
                v_app2 = out2['v_app']
                cam_rv1_down_16 = F.interpolate(cam_rv1_down, size=v_app1.shape[2:],
                                                mode='bilinear', align_corners=True)
                cam_rv2_down_16 = F.interpolate(cam_rv2_down, size=v_app2.shape[2:],
                                                mode='bilinear', align_corners=True)
                proto_app1 = compute_appearance_prototypes(v_app1, cam_rv1_down_16, label, args.bg_threshold)
                proto_app2 = compute_appearance_prototypes(v_app2, cam_rv2_down_16, label, args.bg_threshold)

            # ============ SAPD: Structure prototypes ============
            with torch.no_grad():
                v_str1 = out1['v_str']
                v_str2 = out2['v_str']
                struct_conf1 = out1['struct_conf']
                struct_conf2 = out2['struct_conf']

                cam_rv1_down_str = F.interpolate(cam_rv1_down, size=v_str1.shape[2:],
                                                 mode='bilinear', align_corners=True)
                cam_rv2_down_str = F.interpolate(cam_rv2_down, size=v_str2.shape[2:],
                                                 mode='bilinear', align_corners=True)
                with torch.no_grad():
                    label_bg = label.clone()
                    bg_score_map = torch.ones_like(cam_rv1_down_str[:, 0:1])
                    cam_str_norm = F.relu(cam_rv1_down_str)
                    cam_max_str = torch.max(cam_str_norm.view(N, 21, -1), dim=-1)[0].view(N, 21, 1, 1) + 1e-5
                    cam_str_norm = cam_str_norm / cam_max_str
                    cam_str_norm[:, 0, :, :] = args.bg_threshold
                    pseudo_label1 = cam_str_norm.argmax(dim=1)

                    cam_str_norm2 = F.relu(cam_rv2_down_str)
                    cam_max_str2 = torch.max(cam_str_norm2.view(N, 21, -1), dim=-1)[0].view(N, 21, 1, 1) + 1e-5
                    cam_str_norm2 = cam_str_norm2 / cam_max_str2
                    cam_str_norm2[:, 0, :, :] = args.bg_threshold
                    pseudo_label2 = cam_str_norm2.argmax(dim=1)

                proto_str1 = compute_structure_prototypes(v_str1, struct_conf1, pseudo_label1)
                proto_str2 = compute_structure_prototypes(v_str2, struct_conf2, pseudo_label2)

            # ============ NPCM: Clustering and L_p2p ============
            with torch.no_grad():
                all_protos = torch.cat([proto_fused, proto_fused_2], dim=0)
                cluster_assign = npcm_kmeans.update_and_assign(all_protos)
                cluster_assign_1 = cluster_assign[:21]
                cluster_assign_2 = cluster_assign[21:]

            # Pixel features for contrastive learning
            # For branch 1: use f_proj1 against proto_fused_2 (cross-view)
            f1 = f_proj1
            f1_down = F.interpolate(f1, size=cam_rv1_down.shape[2:], mode='bilinear', align_corners=True)
            N_f, C_f, H_f, W_f = f1_down.shape
            f1_flat = F.normalize(f1_down.permute(0, 2, 3, 1).reshape(-1, C_f), p=2, dim=1)

            with torch.no_grad():
                scores1 = F.softmax(cam_rv1_down * label, dim=1)
                pl1 = scores1.argmax(dim=1).reshape(-1)

            loss_p2p = npcm_p2p_loss(
                f1_flat, pl1, proto_fused_2,
                cluster_assign_2, K=args.npcm_K, tau=args.nce_tau, M=args.npcm_M
            )

            # ============ SAPD: Appearance loss L_app (cross-branch) ============
            v_app1_flat = F.normalize(v_app1.permute(0, 2, 3, 1).reshape(-1, 128), p=2, dim=1)
            v_app2_flat = F.normalize(v_app2.permute(0, 2, 3, 1).reshape(-1, 128), p=2, dim=1)

            with torch.no_grad():
                scores1_app = F.softmax(cam_rv1_down_16 * label, dim=1)
                pl1_app = scores1_app.argmax(dim=1).reshape(-1)
                scores2_app = F.softmax(cam_rv2_down_16 * label, dim=1)
                pl2_app = scores2_app.argmax(dim=1).reshape(-1)

            # cross-branch: branch-1 emb vs branch-2 proto
            loss_app1 = sapd_contrast_loss(v_app1_flat, proto_app2, pl1_app, tau=args.nce_tau)
            loss_app2 = sapd_contrast_loss(v_app2_flat, proto_app1, pl2_app, tau=args.nce_tau)
            loss_app = (loss_app1 + loss_app2) / 2

            # ============ SAPD: Structure loss L_str (cross-branch) ============
            v_str1_flat = F.normalize(v_str1.permute(0, 2, 3, 1).reshape(-1, 128), p=2, dim=1)
            v_str2_flat = F.normalize(v_str2.permute(0, 2, 3, 1).reshape(-1, 128), p=2, dim=1)

            pl1_str = pseudo_label1.reshape(-1)
            pl2_str = pseudo_label2.reshape(-1)

            loss_str1 = sapd_contrast_loss(v_str1_flat, proto_str2, pl1_str, tau=args.nce_tau)
            loss_str2 = sapd_contrast_loss(v_str2_flat, proto_str1, pl2_str, tau=args.nce_tau)
            loss_str = (loss_str1 + loss_str2) / 2

            # ============ Affinity loss L_aff ============
            f_aff1 = out1['cam_rv_down']
            loss_aff1 = affinity_loss(f_aff1, pl1.view(N, H_f, W_f))
            f_aff2 = out2['cam_rv_down']
            _, _, H_aff2, W_aff2 = f_aff2.shape
            loss_aff2 = affinity_loss(f_aff2, pl2_str.view(N, *v_str2.shape[2:]))
            loss_aff = (loss_aff1 + loss_aff2) / 2

            # ============ Total loss ============
            loss = loss_cls + loss_er + loss_ecr + loss_aff + loss_p2p + args.alpha_app * loss_app + args.beta_str * loss_str

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({
                'loss': loss.item(),
                'loss_cls': loss_cls.item(),
                'loss_er': loss_er.item(),
                'loss_ecr': loss_ecr.item(),
                'loss_aff': loss_aff.item() if isinstance(loss_aff, torch.Tensor) else loss_aff,
                'loss_p2p': loss_p2p.item(),
                'loss_app': loss_app.item(),
                'loss_str': loss_str.item(),
            })

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                vals = avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr',
                                     'loss_aff', 'loss_p2p', 'loss_app', 'loss_str')
                print('Iter:%5d/%5d | ' % (optimizer.global_step - 1, max_step),
                      'loss: %.4f | cls: %.4f | er: %.4f | ecr: %.4f | '
                      'aff: %.4f | p2p: %.4f | app: %.4f | str: %.4f'
                      % vals,
                      'imps:%.1f | ' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s | ' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                loss_dict = {
                    'loss': loss.item(),
                    'loss_cls': loss_cls.item(),
                    'loss_er': loss_er.item(),
                    'loss_ecr': loss_ecr.item(),
                    'loss_aff': loss_aff.item() if isinstance(loss_aff, torch.Tensor) else loss_aff,
                    'loss_p2p': loss_p2p.item(),
                    'loss_app': loss_app.item(),
                    'loss_str': loss_str.item(),
                }

                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
                tblogger.add_scalar('mspf/fusion_w0', fusion_w[0].item(), itr)
                tblogger.add_scalar('mspf/fusion_w1', fusion_w[1].item(), itr)
                tblogger.add_scalar('mspf/fusion_w2', fusion_w[2].item(), itr)
                tblogger.add_scalar('mspf/fusion_w3', fusion_w[3].item(), itr)
        else:
            print('')
            timer.reset_stage()

    print(args.session_name)
    torch.save(model.module.state_dict(), args.session_name + '.pth')
