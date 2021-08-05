#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Xiao Qinfeng
# @Date:   2021/8/5 11:21
# @Last Modified by:   Xiao Qinfeng
# @Last Modified time: 2021/8/5 11:21
# @Software: PyCharm

import warnings
from typing import Union, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy import signal


def spectrogram(
        waveform: Tensor,
        pad: int,
        window: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: Optional[float],
        normalized: bool,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        return_complex: bool = True,
) -> Tensor:
    r"""Create a spectrogram or a batch of spectrograms from a raw audio signal.
    The spectrogram can be either magnitude-only or complex.
    Args:
        waveform (Tensor): Tensor of audio of dimension (..., time)
        pad (int): Two sided padding of signal
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        power (float or None): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead.
        normalized (bool): Whether to normalize by magnitude after stft
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy. Default: ``True``
        return_complex (bool, optional):
            Indicates whether the resulting complex-valued Tensor should be represented with
            native complex dtype, such as `torch.cfloat` and `torch.cdouble`, or real dtype
            mimicking complex value with an extra dimension for real and imaginary parts.
            (See also ``torch.view_as_real``.)
            This argument is only effective when ``power=None``. It is ignored for
            cases where ``power`` is a number as in those cases, the returned tensor is
            power spectrogram, which is a real-valued tensor.
    Returns:
        Tensor: Dimension (..., freq, time), freq is
        ``n_fft // 2 + 1`` and ``n_fft`` is the number of
        Fourier bins, and time is the number of window hops (n_frame).
    """
    if power is None and not return_complex:
        warnings.warn(
            "The use of pseudo complex type in spectrogram is now deprecated."
            "Please migrate to native complex type by providing `return_complex=True`. "
            "Please refer to https://github.com/pytorch/audio/issues/1337 "
            "for more details about torchaudio's plan to migrate to native complex type."
        )

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        onesided=onesided,
        return_complex=True,
    )

    # unpack batch
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

    if normalized:
        spec_f /= window.pow(2.).sum().sqrt()
    if power is not None:
        if power == 1.0:
            return spec_f.abs()
        return spec_f.abs().pow(power)
    if not return_complex:
        return torch.view_as_real(spec_f)
    return spec_f
