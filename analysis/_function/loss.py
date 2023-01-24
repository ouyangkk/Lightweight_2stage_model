import torch.nn as nn
import torch
from torch.nn.modules.loss import _Loss

def mse_loss(output, target):
    loss_f2 = nn.MSELoss(size_average=True, reduce=True, reduction='mean' )
    loss = loss_f2(output, target)
    return loss

class Adaptive_weight_loss(nn.Module):
    def __init__(self):
        super(Adaptive_weight_loss, self).__init__()
    def forward(self, weight, obj_speech, ref_specch, res_noise):
        dis_speech_power = torch.mean(torch.mean(torch.pow(obj_speech-ref_specch, 2), dim=-1), dim=-1)
        res_noise_power = torch.mean(torch.mean(torch.pow(res_noise, 2), dim=-1), dim=-1)
        weight_noise = torch.ones_like(weight) - weight
        loss = torch.mean(weight*dis_speech_power + weight_noise*res_noise_power)
        return loss


class MultiSrcNegSDR(_Loss):
    """Base class for computing negative SI-SDR, SD-SDR and SNR for a given
    permutation of source and their estimates.

    Args:
        sdr_type (string): choose between "snr" for plain SNR, "sisdr" for
            SI-SDR and "sdsdr" for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target
            and estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.

    Shape:
        est_targets (:class:`torch.Tensor`): Expected shape [batch, n_src, time].
            Batch of target estimates.
        targets (:class:`torch.Tensor`): Expected shape [batch, n_src, time].
            Batch of training targets.

    Returns:
        :class:`torch.Tensor`: with shape [batch] if reduction='none' else
            [] scalar if reduction='mean'.

    Examples

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(MultiSrcNegSDR("sisdr"),
        >>>                            pit_from='perm_avg')
        >>> loss = loss_func(est_targets, targets)

    References
        - [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.

    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src]
            pair_wise_dot = torch.sum(est_targets * targets, dim=2, keepdim=True)
            # [batch, n_src]
            s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + self.EPS
            # [batch, n_src, time]
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            # [batch, n_src, time]
            scaled_targets = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_targets - targets
        else:
            e_noise = est_targets - scaled_targets
        # [batch, n_src]
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
            torch.sum(e_noise ** 2, dim=2) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -torch.mean(pair_wise_sdr, dim=-1)