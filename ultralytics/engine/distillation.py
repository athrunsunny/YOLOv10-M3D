import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_VALID_FEATURE_LOSS = {"cwd", "mgd", "at", "atm", "skd", "pkd"}
_VALID_LOGIT_LOSS = {'dkd', 'qf'}
# _YOLOV8_DISTILL_LAYER = ["6", "8", "12", "15", "18", "21"]
_YOLOV10_DISTILL_LAYER = ["6", "8", "13", "16", "19", "22"]
# layers = ["2","4","6","8","12","15","18","21"]
_LOSS_FACTOR_CWD = 0.15  #
_LOSS_FACTOR_MGD = 0.03  # pose-s+pose-m 测到最优参数：初始值0.04，权重衰减到0.25
_LOSS_FACTOR_AT = 0.033  # pose-s+pose-m 测到最优参数：初始值0.033，权重衰减到0.1
_LOSS_FACTOR_ATM = 0.04
_LOSS_FACTOR_SKD = 0.25
_LOSS_FACTOR_PKD = 0.15

_LOSS_FACTOR_DKD = 1.0  # 暂未实现
_LOSS_FACTOR_QF = 1.0  # 暂未实现


# -------------------------------------------------
#                     处理函数
# -------------------------------------------------
def feature_norm(x: torch.Tensor) -> torch.Tensor:
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/ppdet/slim/distill_loss.py
    PKD（Pearson）、AT（Attention Transfer）等论文明确强调：
    L2/zero-mean/std 归一化，且对不同 backbone、结构兼容性高。
    """
    n, c, h, w = x.shape
    x = x.permute(1, 0, 2, 3).contiguous().view(c, -1)
    mu = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    x = ((x - mu) / (std + 1e-6)).view(c, n, h, w).permute(1, 0, 2, 3)
    return x


def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# -------------------------------------------------
#                特征级蒸馏 LOSS
# -------------------------------------------------
class CWDFeatureLoss(nn.Module):
    """ Channel-wise Distillation Loss. https://arxiv.org/abs/2011.13256 """

    def __init__(self, tau: float = 1.0, loss_weight: float = 1.0, eps: float = 1e-6):
        """
        Args:
            tau         温度系数 τ (0.5~1.0)
            loss_weight 总权重
            eps         防止 log(0)
        """
        super().__init__()
        self.tau = tau
        self.w = loss_weight
        self.eps = eps
        self.kl = nn.KLDivLoss(reduction='sum')  # 使用 sum 再手动除以 N*C，与论文/Paddle 实现保持一致

    @staticmethod
    def _spatial_flatten(x: torch.Tensor) -> torch.Tensor:
        """(N,C,H,W) → (N*C, HW)"""
        n, c, h, w = x.shape
        return x.flatten(2).reshape(n * c, -1)

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t), "teacher / student 层数不一致"

        loss = 0.
        for fs, ft in zip(y_s, y_t):
            assert fs.shape == ft.shape, "feature shape 不一致"
            n, c, _, _ = fs.shape

            # 展平 & 温度缩放
            ps = self._spatial_flatten(fs) / self.tau
            pt = self._spatial_flatten(ft) / self.tau

            # KL( P_T ‖ P_S )
            log_ps = F.log_softmax(ps, dim=1)
            prob_pt = F.softmax(pt, dim=1).clamp_min(self.eps)

            loss += self.kl(log_ps, prob_pt) * (self.tau ** 2) / (n * c)  # 除 N*C

        return self.w * loss


class MGDFeatureLoss(nn.Module):
    """
    Mask Guided Distillation (MGD) 论文: https://arxiv.org/abs/2105.12968
    严格对齐 Paddle MGD实现，可选normalize/mask/align/generation。
    """

    def __init__(self, channels_s: list, channels_t: list, device=None,
                 alpha_mgd: float = 2e-5,  # 论文单阶段默认 2e-5，双阶段默认5e-7
                 lambda_mgd: float = 0.65,  # 论文单阶段默认 0.65，双阶段默认0.45
                 normalize: bool = True,
                 loss_weight: float = 1.0):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha = alpha_mgd
        self.lmbd = lambda_mgd
        self.normalize = normalize
        self.w = loss_weight
        self.mse_loss = nn.MSELoss(reduction='sum')

        # 为每对特征各建一个生成器 G_i
        self.aligns = nn.ModuleList()  # 用列表，不会梯度回传
        self.generations = nn.ModuleList()

        for s_c, t_c in zip(channels_s, channels_t):
            # align
            if s_c != t_c:
                align = nn.Conv2d(s_c, t_c, 1, 1, 0, bias=False).to(device)
                kaiming_init(align)
            else:
                align = nn.Identity()
            self.aligns.append(align)
            # generation
            gen = nn.Sequential(
                nn.Conv2d(t_c, t_c, 3, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(t_c, t_c, 3, padding=1, bias=False)
            ).to(device)
            gen.apply(kaiming_init)
            self.generations.append(gen)

    def _one_loss(self, feat_s, feat_t, align, generation):
        n = feat_s.shape[0]
        feat_s = align(feat_s)
        # 生成随机mask
        mask = (torch.rand(n, 1, feat_s.shape[2], feat_s.shape[3], device=feat_s.device) < self.lmbd).float()
        # mask+generation
        feat_s = generation(feat_s * mask)
        # normalize（paddle默认开）
        if self.normalize:
            feat_s = feature_norm(feat_s)
            feat_t = feature_norm(feat_t)
        loss = self.mse_loss(feat_s, feat_t.detach()) / n

        return loss

    def forward(self, y_s, y_t, layer=None):
        assert len(y_s) == len(y_t) == len(self.generations)

        losses = []
        for idx in range(len(y_s)):
            fs, ft = y_s[idx], y_t[idx]
            align, gen = self.aligns[idx], self.generations[idx]
            losses.append(self._one_loss(fs, ft, align, gen))

        return self.w * self.alpha * sum(losses) / len(y_s)  # 这里多除一个层数，否则数值太大，可能按维度加权更好


class ATFeatureLoss(nn.Module):
    """ Attention Transfer (attention maps) — Zagoruyko & Komodakis 2017 """

    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-6, normalize: bool = True):
        super().__init__()
        self.w = loss_weight
        self.eps = eps
        self.normalize = normalize

    @staticmethod
    def _attention(f: torch.Tensor, eps: float) -> torch.Tensor:
        """ f : (N, C, H, W)  ->  (N, 1, H*W)  (L2-normed vector) """
        n, c, h, w = f.size()
        att = f.pow(2).mean(dim=1, keepdim=True)  # (N,1,H,W)
        att = att.flatten(2)  # (N,1,HW)
        att = att / (att.norm(p=2, dim=2, keepdim=True) + eps)
        return att

    def forward(self, ys, yt):
        """ ys, yt : List[Tensor] 每层 (N,C,H,W) """
        loss = 0.0
        for feat_s, feat_t in zip(ys, yt):
            if self.normalize:
                feat_s = feature_norm(feat_s)
                feat_t = feature_norm(feat_t)
            As = self._attention(feat_s, self.eps)
            At = self._attention(feat_t, self.eps)
            # 使用 reduction='mean'，与分辨率无关，改为sum使得权重变小
            loss += F.mse_loss(As, At, reduction='sum')
        return loss * self.w


class ATMFeatureLoss(nn.Module):
    """
    自己魔改的attention损失：先小带后小。
    注意力生成最显著特征，那么较小的值是否可能是噪声呢？
    我们可以先把较大的值先减少，这样前面被mask的小值就会相对变大，再带动这些原本较小的值减小。
    AT + Mask (只在教师注意力前 p% 的位置计算 Loss)
    Args:
        keep_ratio: 取多少比例的高分注意力 (0<keep_ratio<=1)
        loss_weight: 最后乘到主损失前的权重 λ
        normalize_feat: 是否先对 raw feature 做 zero-mean/unit-var 归一化
    """

    def __init__(self,
                 keep_ratio: float = 0.5,
                 loss_weight: float = 1.0,
                 eps: float = 1e-6,
                 normalize_feat: bool = True):
        super().__init__()
        assert 0. < keep_ratio <= 1., "`keep_ratio` should be in (0,1]"
        self.p = keep_ratio
        self.w = loss_weight
        self.eps = eps
        self.normalize_feat = normalize_feat
        self.mse = nn.MSELoss(reduction='sum')

    @staticmethod
    def _attention(f: torch.Tensor, eps: float) -> torch.Tensor:
        """ f : (N, C, H, W)  ->  (N, 1, H*W)  (L2-normed vector) """
        n, c, h, w = f.size()
        att = f.pow(2).mean(dim=1, keepdim=True)  # (N,1,H,W)
        att = att.flatten(2)  # (N,1,HW)
        att = att / (att.norm(p=2, dim=2, keepdim=True) + eps)
        return att

    def forward(self, feats_s: list, feats_t: list) -> torch.Tensor:
        assert len(feats_s) == len(feats_t)

        total_loss = 0.0
        for fs, ft in zip(feats_s, feats_t):
            # 0) 归一化原始特征（可选）
            if self.normalize_feat:
                fs = feature_norm(fs)
                ft = feature_norm(ft.detach())
            else:
                ft = ft.detach()

            # 得到注意力图
            As = self._attention(fs, self.eps)  # (N,1,HW)
            At = self._attention(ft, self.eps)  # (N,1,HW)

            diff2 = (As - At).pow(2)  # (N,1,HW)

            if self.p < 0.999:  # p≈1 直接全部参与
                # 每个样本独立取 Top-p%
                k = max(1, int(self.p * diff2.shape[-1]))
                topk, _ = torch.topk(diff2, k=k, dim=-1, largest=True, sorted=False)
                thresh = topk.min(dim=-1, keepdim=True).values  # (N,1,1)
                mask = (diff2 >= thresh)  # bool mask
                diff2 = diff2 * mask.float()

            # 与 AT 保持同量纲：直接 Σ
            total_loss += diff2.sum()

        return self.w * total_loss


class SKDFeatureLoss(nn.Module):
    """
    Spatial-Knowledge-Distillation (Thin-Plate, 2021)(memory-friendly)
    -------------------------------------------------
    * 对每个 feature 先 L2-norm，然后随机采样 k 个空间位置；
    * 在采样点上做空间相关矩阵 S = xᵀx / C  （C 为通道数）；
    * Student / Teacher 的 S 做 MSE。
    这样显存占用 O(k²)，与原论文 (HW)² 比减少几个数量级。
    """

    def __init__(self,
                 k: int = 256,
                 loss_weight: float = 1.0,
                 normalize: bool = True):
        """
        Args:
            k            相关点个数（64 ~ 512 常用）
            loss_weight  总损失权重 λ_kd
            normalize    是否在计算相关前做 L2-norm
        """
        super().__init__()
        self.k = k
        self.w = loss_weight
        self.normalize = normalize
        self.mse = nn.MSELoss(reduction='sum')

    def _corr_sampling(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (N, C, H, W)
        Returns:
            S:   (N, k, k)  — 若 k >= H·W 则为 (N, HW, HW)
        """
        n, c, h, w = feat.shape
        # (N, C, HW)  —>  channel-wise L2-norm
        feat = feat.flatten(2).contiguous()  # (N, C, HW)
        if self.normalize:
            feat = F.normalize(feat, dim=1)
        hw = h * w

        if self.k < hw:  # 随机采样 k 个位置
            idx = torch.randperm(hw, device=feat.device)[: self.k]
            feat = feat[..., idx]  # (N, C, k)
        # 相关矩阵  S = Xᵀ·X / C
        S = torch.matmul(feat.transpose(1, 2), feat) / c  # (N, k, k) or (N, HW, HW)
        return S

    def forward(self, y_s: list, y_t: list) -> torch.Tensor:
        assert len(y_s) == len(y_t), "Student/Teacher 层数不一致"
        layer_losses = []
        for fs, ft in zip(y_s, y_t):
            S_s = self._corr_sampling(fs)
            S_t = self._corr_sampling(ft.detach())
            layer_losses.append(self.mse(S_s, S_t))

        loss = torch.stack(layer_losses).mean()  # 按层平均
        return self.w * loss


class PKDFeatureLoss(nn.Module):
    """PKD: Pearson-Correlation based KD  (CVPR-21)."""

    def __init__(self,
                 normalize: bool = True,
                 loss_weight: float = 1.0,
                 resize_stu: bool = True):
        """
        Args:
            normalize  : 是否执行特征归一化 (zero-mean / unit-var)
            loss_weight: 总系数
            resize_stu : HW 不一致时，是否对 student 做双线性缩放
        """
        super().__init__()
        self.normalize = normalize
        self.w = loss_weight
        self.resize_stu = resize_stu
        self.mse = nn.MSELoss(reduction='mean')

    @staticmethod
    def _feature_norm(x: torch.Tensor) -> torch.Tensor:
        """逐通道零均值 / 单位方差归一化.  Shape (N,C,H,W)."""
        n, c, h, w = x.shape
        x = x.permute(1, 0, 2, 3).contiguous().view(c, -1)  # (C, N*H*W)
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = ((x - mu) / (std + 1e-6)).view(c, n, h, w).permute(1, 0, 2, 3)
        return x

    def _one_loss(self, fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        if fs.shape[2:] != ft.shape[2:]:  # HW 不一致
            if self.resize_stu:
                fs = F.interpolate(fs, ft.shape[2:], mode='bilinear', align_corners=False)
            else:
                ft = F.interpolate(ft, fs.shape[2:], mode='bilinear', align_corners=False)

        if self.normalize:
            fs = self._feature_norm(fs)
            ft = self._feature_norm(ft)

        # 等价于 ½·MSE ≈ 1-ρ  (ρ: Pearson)
        return 0.5 * self.mse(fs, ft)

    def forward(self, feats_s: list, feats_t: list, *_) -> torch.Tensor:
        assert len(feats_s) == len(feats_t)
        loss = sum(self._one_loss(fs, ft.detach()) for fs, ft in zip(feats_s, feats_t))
        return self.w * loss


# -------------------------------------------------
#                FeatureLoss 组合
# -------------------------------------------------
class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', device=None, loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller.lower()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 判断哪些层需要对齐
        self.need_align = [s_c != t_c for s_c, t_c in zip(channels_s, channels_t)]

        # 只为需要对齐的层创建align_module，其余为None
        self.align_module = nn.ModuleList([
            nn.Conv2d(s_c, t_c, kernel_size=1, stride=1, padding=0).to(device) if need else nn.Identity()
            for s_c, t_c, need in zip(channels_s, channels_t, self.need_align)
        ])

        # BatchNorm2d归一化，对齐后归一化，affine为True让其自学
        # self.norm_t = nn.ModuleList([nn.BatchNorm2d(t_c, affine=True).to(device) for t_c in channels_t])  # 对齐后的BN
        # self.norm_s = nn.ModuleList([nn.BatchNorm2d(s_c, affine=True).to(device) for s_c in channels_s])  # 对齐前的BN

        if distiller == 'mgd':
            self.feature_loss = MGDFeatureLoss(channels_s, channels_t, device=device, loss_weight=_LOSS_FACTOR_MGD)
        elif distiller == 'cwd':
            self.feature_loss = CWDFeatureLoss(loss_weight=_LOSS_FACTOR_CWD)
        elif distiller == 'at':
            self.feature_loss = ATFeatureLoss(loss_weight=_LOSS_FACTOR_AT)
        elif distiller == 'atm':
            self.feature_loss = ATMFeatureLoss(loss_weight=_LOSS_FACTOR_ATM)
        elif distiller == 'skd':
            self.feature_loss = SKDFeatureLoss(loss_weight=_LOSS_FACTOR_SKD)
        elif distiller == 'pkd':
            self.feature_loss = PKDFeatureLoss(normalize=False, loss_weight=_LOSS_FACTOR_PKD)
        else:
            raise NotImplementedError(f"{self.distiller} not supported")

    def forward(self, y_s, y_t):
        """
        :param y_s: 学生模型多层输出
        :param y_t: 教师模型多层输出
        :return:   特征蒸馏损失
        """
        assert len(y_s) == len(y_t)
        t_feats, s_feats = [], []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # =============== 特征处理策略分支 ================
            if self.distiller in {'cwd', 'skd', 'pkd'}:
                # CWD/SKD/PKD要求特征通道数完全对齐，如果通道数不同，插入align_module（一般是1x1卷积），否则identity
                s_proc = self.align_module[idx](s)
                # 对齐通道后再做归一化（对齐后特征有变化）,trick:统一用 teacher 的BN参数对齐，再算loss。
                # s_proc = self.norm_t[idx](s_proc)
                # t_proc = self.norm_t[idx](t.detach())
                s_proc = feature_norm(s_proc)
                t_proc = feature_norm(t.detach())
            elif self.distiller in {'mgd', 'at', 'atm'}:
                # MGDFeatureLoss 需要传入原始s, t，内部有自己的通道对齐，这里只用归一化，不再做align_module
                # AT原论文就是直接做L2归一化的注意力图，可以只对teacher归一化
                s_proc = s
                t_proc = t.detach()
                # s_proc = self.norm_s[idx](s_proc)
                # t_proc = self.norm_t[idx](t.detach())
            else:
                raise NotImplementedError(f"不支持的distiller类型: {self.distiller}")

            # 加入特征列表，后面给Loss用
            s_feats.append(s_proc)
            t_feats.append(t_proc)

        loss = self.feature_loss(s_feats, t_feats)
        return self.loss_weight * loss


# -------------------------------------------------
#             logits 级蒸馏 LOSS：DKD、QFD
# -------------------------------------------------
class DKDLogitsLoss(nn.Module):
    def __init__(self, alpha=1., beta=8., tau=4., w=1.0):
        super().__init__()
        self.a, self.b, self.t, self.w = alpha, beta, tau, w
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, s, t, pos_mask=None):
        # s, t: (B, M, C)  or (B, L, C)
        if pos_mask is None:
            pos_mask = torch.ones_like(s[..., :1], dtype=torch.bool, device=s.device)
        s_scaled, t_scaled = s / self.t, t.detach() / self.t
        log_ps, pt = F.log_softmax(s_scaled, -1), F.softmax(t_scaled, -1)
        pos = pos_mask.expand_as(log_ps)
        neg = ~pos
        loss_pos = self.kl(log_ps[pos], pt[pos]) if pos.any() else 0.
        loss_neg = self.kl(log_ps[neg], pt[neg]) if neg.any() else 0.
        return self.w * self.t ** 2 * (self.a * loss_pos + self.b * loss_neg)


class QFKDLoss(nn.Module):
    """ YOLO 头三分支对齐 (obj/cls/box) """

    def __init__(self, w_obj=1.0, w_cls=1.0, w_box=2.0, loss_weight=1.0):
        super().__init__()
        self.w_obj, self.w_cls, self.w_box, self.w = w_obj, w_cls, w_box, loss_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.l1 = nn.SmoothL1Loss(reduction='mean')

    def _split(self, p):
        # p: (B, M, 5+nc)
        box, obj, cls = p.split([4, 1, p.shape[-1] - 5], dim=-1)
        return box, obj, cls

    def forward(self, s_logit, t_logit, **_):
        b_s, o_s, c_s = self._split(s_logit)
        b_t, o_t, c_t = self._split(t_logit.detach())
        loss = (
                self.w_obj * self.bce(o_s, o_t.sigmoid()) +
                self.w_cls * self.bce(c_s, c_t.sigmoid()) +
                self.w_box * self.l1(b_s, b_t)
        )
        return self.w * loss


# -------------------------------------------------
#                LogitLoss 组合
# -------------------------------------------------
class LogitLoss(nn.Module):
    def __init__(self, distiller='dkd', loss_weight=1.0):
        super().__init__()
        name = distiller.lower()
        if name not in _VALID_LOGIT_LOSS:
            raise NotImplementedError(name)
        self.loss_weight = loss_weight
        if name == 'dkd':
            self.loss_fn = DKDLogitsLoss(w=_LOSS_FACTOR_DKD)
        else:  # 'qf'
            self.loss_fn = QFKDLoss(loss_weight=_LOSS_FACTOR_QF)
        self.name = name

    def forward(self, s_logit, t_logit, **extra):
        return self.loss_fn(s_logit, t_logit, **extra) * self.loss_weight


class YOLOv8DistillationLoss:
    """
    用于插入YOLOv8主干训练流程，实现知识蒸馏损失计算与反向传播。
    调用方式：
        1. 初始化时，传入已展开的student和teacher模型（.model）。
        2. 在train时注册hook，收集feature/logit。
        3. 每次前向传播后，调用get_loss()获得蒸馏损失，累加到原始损失中，backward时自动传播到学生模型参数。
    """

    def __init__(self, model_s, model_t, distiller="mgd", device=None):
        self.distiller = distiller.lower()
        # 判定蒸馏类型
        if self.distiller in _VALID_FEATURE_LOSS:
            self._type = "feature"
        elif self.distiller in _VALID_LOGIT_LOSS:
            self._type = "logit"
        else:
            raise NotImplementedError(f"distiller '{distiller}' not supported")

        # 缓存YOLO的任务类型（detect/segment/pose），以及类别数
        self.task = getattr(model_s, 'task', 'detect')
        self.nc = getattr(model_s, 'nc', None)

        # -------- feature-KD（特征蒸馏）初始化 --------
        if self._type == "feature":
            # 遍历指定的蒸馏层，分别收集学生和教师每一层的输出通道数
            def _match(n, lid):
                return f"model.{lid}.cv2.conv" in n

            ch_s, ch_t = [], []
            for lid in _YOLOV10_DISTILL_LAYER:
                for n, m in model_s.named_modules():
                    if _match(n, lid):
                        ch_s.append(m.out_channels)
                        break
                else:
                    raise ValueError(f"Student layer {lid} not found")
                for n, m in model_t.named_modules():
                    if _match(n, lid):
                        ch_t.append(m.out_channels)
                        break
                else:
                    raise ValueError(f"Teacher layer {lid} not found")

            print(f"\033[32mINFO:\033[0m  feature distiller: {self.distiller}")
            print(f"\033[32mINFO:\033[0m  student channels: {ch_s}")
            print(f"\033[32mINFO:\033[0m  teacher channels: {ch_t}")

            # 创建FeatureLoss实例（已详细处理align、BN、内部蒸馏算法）
            assert len(ch_s) == len(ch_t), f"Layer channel count mismatch: {len(ch_s)}, {len(ch_t)}"
            self.D_loss_fn = FeatureLoss(ch_s, ch_t, distiller=self.distiller, device=device)

            # 对应的 hook modules
            self.student_modules = nn.ModuleList()
            self.teacher_modules = nn.ModuleList()
            for lid in _YOLOV10_DISTILL_LAYER:
                for n, m in model_s.named_modules():
                    if _match(n, lid):
                        self.student_modules.append(m)
                        break
                for n, m in model_t.named_modules():
                    if _match(n, lid):
                        self.teacher_modules.append(m)
                        break

        # ------------------ logit-KD ------------------
        else:
            self.D_loss_fn = LogitLoss(distiller=self.distiller, loss_weight=0.5)

            # hook 掉对应 head
            # ultralytics detect head 通常在 model.model[-1].predict 或 .cv2.conv
            head_s = model_s.model[-1].predict if hasattr(model_s.model[-1], 'predict') else model_s.model[-1]
            head_t = model_t.model[-1].predict if hasattr(model_t.model[-1], 'predict') else model_t.model[-1]
            self.student_modules = nn.ModuleList([head_s])
            self.teacher_modules = nn.ModuleList([head_t])

        # 统一缓存 & hook handle
        self.student_feats, self.teacher_feats, self._handles = [], [], []

    @property
    def distill_type(self):
        return self._type

    def register_hook(self):
        """
        在每一轮训练/验证开始前调用一次，注册forward hook，
        把指定层的输出结果收集到student_feats/teacher_feats里。
        必须在forward之前注册，防止漏采样。
        """
        self.remove_handle_()
        # feature 需要同时 hook student/teacher 多个层
        # logit   只 hook head，一样塞进 list
        for s_mod, t_mod in zip(self.student_modules, self.teacher_modules):
            self._handles.append(s_mod.register_forward_hook(self._hook(self.student_feats)))
            self._handles.append(t_mod.register_forward_hook(self._hook(self.teacher_feats)))

    def _hook(self, buff):
        """ 用于注册的hook函数，把每次forward的特征append到缓存列表。 """

        def fn(_, __, out):
            buff.append(out)

        return fn

    def remove_handle_(self):
        """ 训练完成/不再需要蒸馏时，移除所有hook，释放显存，防止内存泄漏。 """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self, *args, **kwargs):
        """
        在每个batch forward后调用一次，自动根据类型计算feature或logit级蒸馏损失，返回给主损失累加。
        - 会自动清空缓存的hook特征。
        - 返回的loss已经是可backward的张量，会对学生网络产生梯度影响。
        """
        if self._type == "feature":
            loss = self.D_loss_fn(self.student_feats, self.teacher_feats)
            # 清空缓存
            self.student_feats.clear()
            self.teacher_feats.clear()
            return loss

        # -------- Logit-KD 分支 --------
        # 拿出 head 的 raw outputs
        raw_s = self.student_feats.pop(0)
        raw_t = self.teacher_feats.pop(0)
        self.student_feats.clear();
        self.teacher_feats.clear()

        # 如果是 tuple/list，取第一个 tensor
        if isinstance(raw_s, (tuple, list)):
            raw_s, raw_t = raw_s[0], raw_t[0]

        B, C, *HW = raw_s.shape

        # 根据任务 reshape → (B, N, C_head)
        if self.task == 'segment':
            # raw: (B, nC, H, W) → (B, H*W, nC)
            nC, H, W = C, HW[0], HW[1]
            s = raw_s.view(B, nC, -1).permute(0, 2, 1)
            t = raw_t.view(B, nC, -1).permute(0, 2, 1)

        elif self.task == 'pose':
            # raw: (B, K, H, W) heatmap → (B, H*W, K)
            K, H, W = C, HW[0], HW[1]
            s = raw_s.view(B, K, -1).permute(0, 2, 1)
            t = raw_t.view(B, K, -1).permute(0, 2, 1)

        else:  # detect
            # raw: (B, M*(5+nc), H, W)
            H, W = HW
            per = C // (H * W)  # per = 5+nc
            N = H * W
            s = raw_s.view(B, N, per)
            t = raw_t.view(B, N, per)

        # 3) DKD 还需要 pos_mask
        extra = {}
        if isinstance(self.D_loss_fn, LogitLoss) and self.D_loss_fn.name == 'dkd':
            # objectness 在第 5 个通道
            pos_mask = (t[..., 4:5].sigmoid() > 0.5)
            extra['pos_mask'] = pos_mask

        # 4) 调用 LogitLoss
        return self.D_loss_fn(s, t, **extra)

        # ------------------------- 获取不同算法的衰减权重 ----------------

    @staticmethod
    def cosine_anneal(epoch: int, total_epochs: int, final_val: float) -> float:
        """
        余弦退火系数，从 1 线性衰减到 final_val。
        epoch:        当前轮次 (0 <= epoch <= total_epochs)
        total_epochs: 总轮次
        final_val:    衰减的最终值 (0 < final_val < 1)
        """
        # cos(pi * epoch / total) 从 cos(0)=1 变到 cos(pi)=–1
        # (1 + cos(...)) / 2 从 1 变到 0
        return final_val + 0.5 * (1.0 - final_val) * (1.0 + math.cos(math.pi * epoch / total_epochs))

    @staticmethod
    def exp_anneal(epoch: int, total_epochs: int, final_val: float, decay_rate: float = 5.0) -> float:
        """
        指数退火系数，从 1 → final_val。

        Parameters:
        - epoch: 当前轮次 (0 ≤ epoch ≤ total_epochs)
        - total_epochs: 总轮次
        - final_val: 衰减后的最终值 (0 < final_val < 1)
        - decay_rate: 衰减速度因子，越大前期越平缓、后期越陡

        Returns:
        - weight: 衰减权重
        """
        t = epoch / total_epochs
        # math.exp(-decay_rate * t) 从 1 → e^{-decay_rate}（通常很小）
        # 归一化后再加上 final_val 保证末尾落在 final_val
        return final_val + (1.0 - final_val) * math.exp(-decay_rate * t)

    def get_kd_weight(self, epoch: int, total_epochs: int) -> float:
        """ 根据当前 epoch 和总 epoch，为不同的蒸馏算法返回对应的权重 λ。 """
        # ------ feature-level methods ------
        if self._type == "feature":
            if self.distiller == "at":
                # 衰减过慢，且后续基本不减少
                return self.cosine_anneal(epoch, total_epochs, final_val=0.1)

            elif self.distiller == "atm":
                # 衰减过慢，且后续基本不减少
                return self.cosine_anneal(epoch, total_epochs, final_val=0.25)

            elif self.distiller == "cwd":
                # 总体会一直衰减，只需最后gj 增加点衰减就行
                return self.cosine_anneal(epoch, total_epochs, final_val=0.5)

            elif self.distiller == "mgd":
                # 先快速减小，再缓慢上升，所以希望权重先缓慢减小，再快速减小
                return self.cosine_anneal(epoch, total_epochs, final_val=0.5)
                # return self.exp_anneal(epoch, total_epochs, final_val=0.1, decay_rate=5.0)

            elif self.distiller == "skd":
                # 同mgd，但是效果提升不明显，所以降低最终权重值
                return self.cosine_anneal(epoch, total_epochs, final_val=0.25)

            elif self.distiller == "pkd":
                # 类似cwd，但是起始提升大，后续小，所以换指数衰减
                return self.cosine_anneal(epoch, total_epochs, final_val=0.5)

            else:
                return 1.0

        return 1.0


if __name__ == '__main__':
    from ultralytics import YOLO

    model_s_path = ""
    model_t_path = ""

    model_s = YOLO(model_s_path)
    model_t = YOLO(model_t_path)

    distill = YOLOv8DistillationLoss(model_s.model, model_t.model, distiller="CWD")
