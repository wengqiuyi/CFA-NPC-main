import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.res2net_v1b_base import Res2Net_model


###############################################################################
##  Basic Building Blocks
###############################################################################
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


###############################################################################
##  FFT-based Cross-Modal Attention (FFT-CMA / FSCM)  — Level-1 fusion X0
###############################################################################
class FSCM(nn.Module):
    """
    Cross-modal attention in the frequency domain.
    The two streams (T1-Block1, T2-Block1) exchange amplitude information
    through element-wise multiplication, then are reconstructed by IFFT
    and concatenated for the final fusion.

    NOTE: This module is equivalent to a Frequency-domain Spectral
          Cross-Modal module (FSCM): both perform spectrum decomposition
          (FFT), cross-modal amplitude modulation, and reconstruction (IFFT).
          The two names refer to the same operator.
    """
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FSCM, self).__init__()
        self.reduce1 = BasicConv2d(in_channels1, out_channels, 1)
        self.reduce2 = BasicConv2d(in_channels2, out_channels, 1)

        # Amplitude attention: weight the amplitude of one stream by the other
        self.amp_att = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.fuse = BasicConv2d(out_channels * 2, out_channels, 3, padding=1)

    def forward(self, x1, x2):
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)

        # Frequency domain decomposition
        fft1 = torch.fft.rfft2(x1, norm='ortho')
        fft2 = torch.fft.rfft2(x2, norm='ortho')
        amp1, pha1 = torch.abs(fft1), torch.angle(fft1)
        amp2, pha2 = torch.abs(fft2), torch.angle(fft2)

        # Cross-modal amplitude modulation
        amp1_swap = amp1 * self.amp_att(amp2)
        amp2_swap = amp2 * self.amp_att(amp1)

        # Reconstruct the complex spectrum
        real1 = amp1_swap * torch.cos(pha1)
        imag1 = amp1_swap * torch.sin(pha1)
        fft1_swap = torch.complex(real1, imag1)

        real2 = amp2_swap * torch.cos(pha2)
        imag2 = amp2_swap * torch.sin(pha2)
        fft2_swap = torch.complex(real2, imag2)

        # Back to spatial domain
        out1 = torch.fft.irfft2(fft1_swap, s=x1.shape[-2:], norm='ortho')
        out2 = torch.fft.irfft2(fft2_swap, s=x2.shape[-2:], norm='ortho')

        out = self.fuse(torch.cat([out1, out2], dim=1))
        return out


# Alias — FFTCMA is identical to FSCM (Frequency-domain Spectral Cross-Modal).
FFTCMA = FSCM


###############################################################################
##  Global Fusion (GF) — Level-2/3/4 fusion X1/X2/X3
###############################################################################
class GlobalFusion(nn.Module):
    """
    Gate-based global fusion. Two features are concatenated, then a soft-max
    gate decides the contribution of each stream at every spatial location.
    """
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(GlobalFusion, self).__init__()
        self.reduce1 = BasicConv2d(in_channels1, out_channels, 1)
        self.reduce2 = BasicConv2d(in_channels2, out_channels, 1)
        self.gate    = nn.Sequential(
            nn.Conv2d(out_channels * 2, 2, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        att = self.gate(torch.cat([x1, x2], dim=1))
        return x1 * att[:, 0:1] + x2 * att[:, 1:2]


###############################################################################
##  Edge Block (Edge 0/1/2/3) — Top-branch feature refinement
###############################################################################
class EdgeBlock(nn.Module):
    def __init__(self, channels):
        super(EdgeBlock, self).__init__()
        self.refine = nn.Sequential(
            BasicConv2d(channels, channels, 3, padding=1),
            BasicConv2d(channels, channels, 3, padding=1),
        )
        self.edge_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        r = self.refine(x)
        g = self.edge_gate(r)
        return x + r * g

###############################################################################
##  Cross-level Fusion Skip (CF) — simple skip connection (fallback)
###############################################################################
class CFSkip(nn.Module):
    """Simple skip projection (kept for compatibility)."""
    def __init__(self, in_channels, out_channels):
        super(CFSkip, self).__init__()
        self.conv = BasicConv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


###############################################################################
##  EFC — Edge-Enhanced Feature Concat (创新点 2)
##      x    ─► pre ─► ⊗  ◄── EdgeDetection(skip)
##                       │
##                       ├──cat── skip ──► CA ──► ⊗ ──► SA ──► ⊗(skip) ──► output
###############################################################################
class EFC(nn.Module):
    """
    Edge-Enhanced Feature Concat module (创新点 2).

    Pipeline (see the figure in the paper):
        1. pre(x)   ──► *   (element-wise multiply with edge(skip))
        2. cat(*output, skip)  ──► channel-attention (CA) ──► *
        3. *output  ──► spatial-attention (SA) ──► *
        4. final   = skip ⊗ sa_mask + proj([pre(x), edge_feat])

    Args:
        channels (int):  channel number of both ``x`` and ``skip``.
        ca_ratio  (int):  bottleneck ratio of the channel attention MLP.
    """
    def __init__(self, channels, ca_ratio=16):
        super(EFC, self).__init__()

        # 1) pre-processing of the main feature x
        self.pre = BasicConv2d(channels, channels, 3, padding=1)

        # 2) edge detection on the skip feature (produces a 0-1 edge map)
        self.edge_det = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        cat_channels = channels * 2
        hidden = max(cat_channels // ca_ratio, 4)

        # 3) channel attention on the cat output
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cat_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, cat_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 4) spatial attention on the channel-refined feature
        self.spatial_att = nn.Sequential(
            nn.Conv2d(cat_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # 5) residual injection of the main branch x / edge_feat into the output
        self.residual_proj = BasicConv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x, skip):
        # step 1:  pre(x) ⊗ edge(skip)
        pre_x = self.pre(x)
        edge  = self.edge_det(skip)
        edge_feat = pre_x * edge

        # step 2:  cat with the original skip
        cat_feat = torch.cat([edge_feat, skip], dim=1)        # 2C

        # step 3:  channel attention (gating)
        ca  = self.channel_att(cat_feat)
        ca_feat = cat_feat * ca

        # step 4:  spatial attention, then apply to skip
        sa  = self.spatial_att(ca_feat)
        gated_skip = skip * sa

        # step 5:  直接将主分支特征作为残差注入
        residual = self.residual_proj(torch.cat([pre_x, edge_feat], dim=1))
        out = gated_skip + residual

        return out
# Backward-compat alias (the module used to be named EFA in this file).
EFA = EFC


###############################################################################
##  CFANet — Cross-level Feature Aggregation Network
##      * Dual-stream encoder (T1, T2)
##      * Three parallel decoder branches (top, middle, bottom)
##      * Returns (mask1, mask2, mask3, mask)
###############################################################################
class CFANet(nn.Module):
    def __init__(self, channel=64, dual_backbone=True, in_channels=3):
        """
        Args:
            channel (int): base channel width for all branches.
            dual_backbone (bool):
                True  -> two independent Res2Net backbones for T1 and T2
                         (suitable for true cross-modal input).
                False -> a single shared backbone, T1 == T2 == x
                         (suitable for single-image input).
            in_channels (int):
                Number of input channels per modality.  For the standard 2.5D
                stack this is ``3`` (centre ± 1 slice).  For pseudo-3D input
                pass an odd integer ``k_slice`` in {5, 7, 9, ...} to take a
                wider z context.  The first-conv weights are automatically
                interpolated from the 3-ch ImageNet checkpoint in
                ``Res2Net_model(..., in_channels)``.
        """
        super(CFANet, self).__init__()
        self.dual_backbone = bool(dual_backbone)
        self.in_channels = int(in_channels)

        # ---- Dual backbone encoders ----
        if dual_backbone:
            self.backbone_t1 = Res2Net_model(50, in_channels=in_channels)
            self.backbone_t2 = Res2Net_model(50, in_channels=in_channels)
        else:
            shared = Res2Net_model(50, in_channels=in_channels)
            self.backbone_t1 = shared
            self.backbone_t2 = shared

        # ---- Cross-modal fusion at the 4 encoder levels ----
        # Res2Net-50 channel widths: x1=256, x2=512, x3=1024, x4=2048
        self.fft_cma_x0 = FFTCMA(256,  256,  channel)      # T1/T2 Block1 -> X0
        self.fft_cma_x1 = FFTCMA(512,  512,  channel)      # T1/T2 Block2 -> X1
        self.gf_x2      = GlobalFusion(1024, 1024, channel)  # T1/T2 Block3 -> X2
        self.gf_x3      = GlobalFusion(2048, 2048, channel)  # T1/T2 Block4 -> X3

        # X0 -> F1 (initial fusion of the top branch)
        self.fusion_x0 = BasicConv2d(channel, channel, 3, padding=1)

        # ---- Top branch : F1 -> Edge0 -> F2 -> Edge1 -> F3 -> Edge2 -> F4 -> Edge3 -> F5 -> mask1
        self.edge0  = EdgeBlock(channel)
        self.edge1  = EdgeBlock(channel)
        self.edge2  = EdgeBlock(channel)
        self.edge3  = EdgeBlock(channel)
        self.head1  = nn.Conv2d(channel, 1, kernel_size=1)

        # ---- Middle branch : F11 -> F12 -> F13 -> F14 -> F15 -> mask2
        #   F11 = X1 + CF(F1)                            (CF = simple skip)
        #   F1i = EFC(F1(i-1), Fi)   for i = 2..5        (EFC = BAM-replacement)
        self.cf_mid1     = CFSkip(channel, channel)     # CF for F11
        self.efc_mid1    = EFC(channel)                 # EFC(F11, F2) -> F12
        self.efc_mid2    = EFC(channel)                 # EFC(F12, F3) -> F13
        self.efc_mid3    = EFC(channel)                 # EFC(F13, F4) -> F14
        self.efc_mid4    = EFC(channel)                 # EFC(F14, F5) -> F15
        self.head2       = nn.Conv2d(channel, 1, kernel_size=1)

        # ---- Bottom branch : F21 -> F22 -> F23 -> F24 -> F25 -> mask3
        #   F21 = X2 + CF(X3)                            (CF = simple skip)
        #   F2i = EFC(F2(i-1), F1(i-1))   for i = 2..5  (EFC = BAM-replacement)
        self.cf_bot1     = CFSkip(channel, channel)     # CF for F21
        self.efc_bot1    = EFC(channel)                 # EFC(F21, F12) -> F22
        self.efc_bot2    = EFC(channel)                 # EFC(F22, F13) -> F23
        self.efc_bot3    = EFC(channel)                 # EFC(F23, F14) -> F24
        self.efc_bot4    = EFC(channel)                 # EFC(F24, F15) -> F25
        self.head3       = nn.Conv2d(channel, 1, kernel_size=1)

        # ---- Final fusion head ----
        # Learn the final prediction from the three decoder branches in
        # feature space so train / val / test can all use the same output.
        self.final_fuse = BasicConv2d(channel * 3, channel, 3, padding=1)
        self.head_final = nn.Conv2d(channel, 1, kernel_size=1)

    # ------------------------------------------------------------------ #
    def _upsample_to(self, x, ref):
        """Bilinearly upsample ``x`` to the spatial size of ``ref``."""
        return F.interpolate(x, size=ref.shape[-2:],
                             mode='bilinear', align_corners=False)

    # ------------------------------------------------------------------ #
    def forward(self, x1, x2=None):
        """
        Args:
            x1 (Tensor): first  input image  (N, C, H, W); C = in_channels
            x2 (Tensor): second input image  (N, C, H, W). If None, x2 = x1.
        Returns:
            mask1, mask2, mask3, mask  (all logits, N x 1 x H x W)
        """
        shared_single_input = (x2 is None) and (not self.dual_backbone)
        if x2 is None:
            x2 = x1
        H, W = x1.shape[-2:]

        # ---- Encoder features ----
        _, t1_1, t1_2, t1_3, t1_4 = self.backbone_t1(x1)
        if shared_single_input:
            t2_1, t2_2, t2_3, t2_4 = t1_1, t1_2, t1_3, t1_4
        else:
            _, t2_1, t2_2, t2_3, t2_4 = self.backbone_t2(x2)

        # ---- 4-level cross-modal fusion ----
        # X0 from FFT-CMA (shallowest, two streams at the highest resolution)
        x0 = self.fft_cma_x0(t1_1, t2_1)                                             # X0
        x1_fused = self.fft_cma_x1(t1_2, t2_2)                                       # X1
        x2_fused = self.gf_x2(t1_3, t2_3)                                            # X2
        x3_fused = self.gf_x3(t1_4, t2_4)                                            # X3

        # =================== Top branch (X0 -> mask1) =================== #
        f1 = self.fusion_x0(x0)
        f2 = self.edge0(f1)
        f3 = self.edge1(f2)
        f4 = self.edge2(f3)
        f5 = self.edge3(f4)

        # ================= Middle branch (X1 -> mask2) ================== #
        # F11 = X1 + CF(F1)                       (CF = simple skip)
        # F1i = EFC(F1(i-1), Fi)   for i=2..5    (EFC = BAM-replacement, 创新点 2)
        f1_d1 = self._upsample_to(self.cf_mid1(f1), x1_fused)
        f11 = x1_fused + f1_d1
        f12 = self.efc_mid1(f11, self._upsample_to(f2, x1_fused))
        f13 = self.efc_mid2(f12, self._upsample_to(f3, x1_fused))
        f14 = self.efc_mid3(f13, self._upsample_to(f4, x1_fused))
        f15 = self.efc_mid4(f14, self._upsample_to(f5, x1_fused))

        # ================= Bottom branch (X2/X3 -> mask3) =============== #
        # F21 = X2 + CF(X3)                                  (CF = simple skip)
        # F2i = EFC(F2(i-1), F1(i-1))   for i=2..5          (EFC = BAM-replacement)
        x3_u2 = self._upsample_to(x3_fused, x2_fused)
        f21 = x2_fused + self.cf_bot1(x3_u2)
        f22 = self.efc_bot1(f21, self._upsample_to(f12, x2_fused))
        f23 = self.efc_bot2(f22, self._upsample_to(f13, x2_fused))
        f24 = self.efc_bot3(f23, self._upsample_to(f14, x2_fused))
        f25 = self.efc_bot4(f24, self._upsample_to(f15, x2_fused))

        # ---- Prediction heads (upsample to input size) ----
        mask1 = F.interpolate(self.head1(f5),  size=(H, W), mode='bilinear', align_corners=False)
        mask2 = F.interpolate(self.head2(f15), size=(H, W), mode='bilinear', align_corners=False)
        mask3 = F.interpolate(self.head3(f25), size=(H, W), mode='bilinear', align_corners=False)

        # ---- Final fused mask ----
        f15_u = self._upsample_to(f15, f5)
        f25_u = self._upsample_to(f25, f5)
        final_feat = self.final_fuse(torch.cat([f5, f15_u, f25_u], dim=1))
        mask = F.interpolate(self.head_final(final_feat), size=(H, W),
                             mode='bilinear', align_corners=False)

        return mask1, mask2, mask3, mask


# ---------------------------------------------------------------------------
# Helper: remap a 3-channel checkpoint trained with in_channels=3 onto an
# arbitrary k_slice > 3 network.  We only need to fix the two backbone
# conv1.0.weight tensors (T1 + T2).  All later conv/bn/fusion layers are
# independent of input channel count and transfer directly.
# Used by train.py --pretrain_ckpt to warm-start k_slice=9 / k_slice=15 runs
# from the existing epoch100 2.5D (k=3) Cod_best_fg checkpoint.
# ---------------------------------------------------------------------------
def remap_ckpt_in_channels(ckpt_state: dict, in_channels_new: int,
                           in_channels_old: int = 3) -> dict:
    """Return a state_dict ready to be loaded into a CFANet with k_slice>3.

    Parameters
    ----------
    ckpt_state : dict
        state_dict from a 3-channel checkpoint (as saved by train.py).
    in_channels_new : int
        Target ``k_slice`` (the new input channels per modality). Must be >= 1.
    in_channels_old : int
        Usually 3 (the legacy 2.5D stack).

    Returns
    -------
    dict
        New state dict.  Only ``backbone_t1.conv1.0.weight`` and
        ``backbone_t2.conv1.0.weight`` are modified; every other tensor is
        a reference to the input (no copies).
    """
    import torch as _torch
    out = dict(ckpt_state)  # shallow copy
    for bb_key in ("backbone_t1.conv1.0.weight", "backbone_t2.conv1.0.weight"):
        if bb_key not in out:
            continue
        w_old = out[bb_key]  # (Cout, 3, k, k)
        if w_old.ndim != 4 or w_old.shape[1] != in_channels_old:
            continue
        if in_channels_new == in_channels_old:
            continue
        Cout, Cin_old, kh, kw = w_old.shape
        flat = w_old.reshape(Cout, Cin_old, kh * kw).permute(0, 2, 1).contiguous()
        scaled = _torch.nn.functional.interpolate(
            flat.float(), size=in_channels_new, mode='linear', align_corners=False
        )
        w_new = scaled.permute(0, 2, 1).reshape(Cout, in_channels_new, kh, kw).contiguous()
        # Re-scale magnitude so total energy per output channel matches the
        # 3-ch case (avoids exploding activations for k>3).
        w_new = w_new * (float(Cin_old) / float(in_channels_new))
        out[bb_key] = w_new.to(dtype=w_old.dtype)
    return out
