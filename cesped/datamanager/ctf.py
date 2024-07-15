import functools

import einops
import torch
import numpy as np


def _compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None):
    '''
    Compute the 2D CTF

    Input:
        freqs (Tensor) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    '''
    assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10 ** 7
    dfang = dfang * torch.pi / 180
    phase_shift = phase_shift * torch.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / (volt + 0.97845e-6 * volt ** 2) ** .5
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = torch.atan2(y, x)
    s2 = x ** 2 + y ** 2
    df = .5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
    gamma = 2 * torch.pi * (-.5 * df * lam * s2 + .25 * cs * lam ** 3 * s2 ** 2) - phase_shift
    ctf = (1 - w ** 2) ** .5 * torch.sin(gamma) - w * torch.cos(gamma)
    if bfactor is not None:
        ctf *= torch.exp(-bfactor / 4 * s2)
    return ctf

@functools.lru_cache(1)
def _get2DFreqs(imageSize, sampling_rate, device=None):
    freqs1d = torch.fft.fftshift(torch.fft.fftfreq(imageSize))
    freqs = torch.stack(torch.meshgrid(freqs1d, freqs1d, indexing='ij'), -1)\
            / sampling_rate
    freqs = freqs.reshape(-1,2)
    if device is not None:
        freqs = freqs.to(device)
    return freqs

def compute_ctf(image_size, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor, device):
    freqs = _get2DFreqs(image_size, sampling_rate, device=device)

    ctf = _compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor)
    ctf = einops.rearrange(ctf, "... (s0 s1) -> ... s0 s1", s0=image_size, s1=image_size)

    return -ctf

def correct_ctf(image, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None, mode='phase_flip',
              wiener_parameter=0.15):
    '''
    Apply the 2D CTF through a Wiener filter

    Input:
        image (Tensor) the BxDxD image in real space
        sampling_rate: in A/pixel
        dfu (float or Bx1 tensor): DefocusU (Angstrom). Positive for underfocus
        dfv (float or Bx1 tensor): DefocusV (Angstrom). Positive for underfocus
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
        mode (string): CTF correction: 'phase_flip' or 'wiener'
        wiener_parameter (float): wiener parameter for not dividing by zero between 0.05 - 0.2 #TODO: Wiegner parameter could also be a per-frequency weight

    '''

    ctf = compute_ctf(image.shape[-1], sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor, device=image.device)

    fimage = torch.fft.fftshift(torch.fft.fft2(image), dim=(-2,-1))

    if mode == 'phase_flip':
        fimage_corrected = fimage * torch.sign(ctf)
    elif mode == 'multiply':
        fimage_corrected = fimage * ctf
    elif mode == 'wiener':
        fimage_corrected = fimage/(ctf + torch.sign(ctf)*wiener_parameter)
    else:
        raise ValueError("Only phase_flip, multiply and wiener are valid")

    image_corrected = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fimage_corrected, dim=(-2,-1))))
    return ctf, image_corrected

