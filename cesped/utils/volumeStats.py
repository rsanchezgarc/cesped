import os

import mrcfile
import numpy as np
from scipy.stats import pearsonr

from cesped.utils.fsc import getFSCResolution


def _loadIfFname(volOrFname):
    if isinstance(volOrFname, (str, os.PathLike)):
        with mrcfile.open(volOrFname) as f:
            vol = f.data.copy()
            samplingRate = float(f.voxel_size.x)
    else:
        vol = volOrFname
        samplingRate = None
    return vol, samplingRate


def compute_stats(gtVolOrFname, predVolOrFname, maskOrFname=None, samplingRate=None, resolution_threshold=0.143):
    gtData, gtSr = _loadIfFname(gtVolOrFname)
    predData, predSr = _loadIfFname(predVolOrFname)
    if maskOrFname is not None:
        mask, maskSr = _loadIfFname(maskOrFname)
    else:
        mask = None
        maskSr = None

    sr = [gtSr, predSr, maskSr]
    if any(sr):
        sr_set = set([round(x,3) for x in sr if x is not None])
        assert len(sr_set) == 1, f"Error, different sampling rates {sr}"
        sr_ = sr_set.pop()
        if samplingRate is not None:
            assert round(sr_, 3) == round(samplingRate, 3), f"Error, different sampling rates {sr_} vs {samplingRate}"
        else:
            samplingRate = sr_
    else:
        assert samplingRate is not None, "Error, sampling rate was not provided"

    corr = pearsonr(gtData.flatten(), predData.flatten()).statistic
    if mask is not None:
        m_corr = pearsonr(gtData.flatten() * mask.flatten(), predData.flatten() * mask.flatten()).statistic
    else:
        m_corr = np.nan

    fscResolution = getFSCResolution(gtData, predData, mask=mask, samplingRate=samplingRate,
                                     resolution_threshold=resolution_threshold)
    return (corr, m_corr), fscResolution


# TODO: REMOVE THIS HARDCODED TEST
if __name__ == "__main__":
    # gtVolFname = "/tmp/chimera/half0/relion_reconstruct_half1.mrc"
    # predVolFname = "/tmp/chimera/half1/relion_reconstruct_half1.mrc"
    # maskFname = "/tmp/chimera/half0/mask.mrc"
    gtVolFname = "/home/sanchezg/cryo/data/preAlignedParticles/NEURIPS_RESULTS/reconstructions/10374/gt.mrc"
    predVolFname = "/home/sanchezg/cryo/data/preAlignedParticles/NEURIPS_RESULTS/reconstructions/10374/locally_refined.mrc"
    maskFname = "/home/sanchezg/cryo/data/preAlignedParticles/NEURIPS_RESULTS/reconstructions/10374/10374_mask.mrc"

    out = compute_stats(gtVolFname, predVolFname, maskFname)
    print(out)
