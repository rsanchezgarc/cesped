import os

import mrcfile
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
        assert len(set([round(x,3) for x in sr if x is not None])) == 1, "Error, different sampling rates"
        if samplingRate is not None:
            assert sr[0] == samplingRate, "Error, different sampling rates"
        else:
            samplingRate = sr[0]
    else:
        assert samplingRate is not None, "Error, sampling rate was not provided"

    corr = pearsonr(gtData.flatten(), predData.flatten()).statistic
    fscResolution = getFSCResolution(gtData, predData, mask=mask, samplingRate=samplingRate,
                                     resolution_threshold=resolution_threshold)
    return corr, fscResolution


# TODO: REMOVE THIS BAD TEST
if __name__ == "__main__":
    gtVolFname = "/tmp/chimera/half0/relion_reconstruct_half1.mrc"
    predVolFname = "/tmp/chimera/half1/relion_reconstruct_half1.mrc"
    maskFname = "/tmp/chimera/half0/mask.mrc"
    out = compute_stats(gtVolFname, predVolFname, maskFname)
    print(out)
