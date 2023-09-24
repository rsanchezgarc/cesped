import argparse
import os.path as osp
import mrcfile
import numpy as np


def getMaskCorrectedFSC(array1, array2, mask, samplingRate,
                        nIter=1, rndThr=0.8, rndOffset=4):

    unmaskedArray1 = np.fft.fftshift(np.fft.fftn(array1))
    unmaskedArray2 = np.fft.fftshift(np.fft.fftn(array2))

    if mask is not None:
        maskedArray1 = np.fft.fftshift(np.fft.fftn(array1*mask))
        maskedArray2 = np.fft.fftshift(np.fft.fftn(array2*mask))

    iSize = array1.shape[0]
    iSize2 = int(iSize/2)

    xV = np.arange(iSize) - iSize2
    XXmesh = np.meshgrid(xV, xV, xV, copy=False, indexing='ij')

    radii = np.zeros((iSize,)*3)

    for k in range(3):
        radii += XXmesh[k]**2
    radii = np.sqrt(radii)

    nShells = iSize2 + 1
    shellList = [None]*nShells

    for k in range(nShells):
        shellList[k] = np.where((radii >= k) * (radii < (k+1)))

    unmaskedCCF = np.real(unmaskedArray1 * unmaskedArray2.conj())
    unmaskedInt1 = np.real(unmaskedArray1 * unmaskedArray1.conj())
    unmaskedInt2 = np.real(unmaskedArray2 * unmaskedArray2.conj())

    if mask is not None:
        maskedCCF = np.real(maskedArray1 * maskedArray2.conj())
        maskedInt1 = np.real(maskedArray1 * maskedArray1.conj())
        maskedInt2 = np.real(maskedArray2 * maskedArray2.conj())

    unmaskedCCF_1D = np.zeros(nShells)
    unmaskedInt1_1D = np.zeros(nShells)
    unmaskedInt2_1D = np.zeros(nShells)

    if mask is not None:
        maskedCCF_1D = np.zeros(nShells)
        maskedInt1_1D = np.zeros(nShells)
        maskedInt2_1D = np.zeros(nShells)

    for k in range(nShells):
        unmaskedCCF_1D[k] = unmaskedCCF[shellList[k]].sum()
        unmaskedInt1_1D[k] = unmaskedInt1[shellList[k]].sum()
        unmaskedInt2_1D[k] = unmaskedInt2[shellList[k]].sum()
        if mask is not None:
            maskedCCF_1D[k] = maskedCCF[shellList[k]].sum()
            maskedInt1_1D[k] = maskedInt1[shellList[k]].sum()
            maskedInt2_1D[k] = maskedInt2[shellList[k]].sum()

    # FSC profiles
    unmaskedFSC = unmaskedCCF_1D/np.sqrt(unmaskedInt1_1D * unmaskedInt2_1D)
    maskedFSC = None
    correctedFSC = None
    phaseRndFSC = None
    if mask is not None:
        maskedFSC = maskedCCF_1D/np.sqrt(maskedInt1_1D * maskedInt2_1D)

        # Index for phase rondimization
        rndIdx = np.where(unmaskedFSC < rndThr)[0]

        if len(rndIdx) == 0:
            rndIdx = nShells - 1
        elif rndIdx[0] > 0:
            rndIdx = rndIdx[0]
        else:
            rndIdx = rndIdx[1]

        print(f'Phase randomization at pixel {rndIdx} =\
              {(iSize*samplingRate)/rndIdx:.2g} Å')

        rndMask = radii >= rndIdx
        nRnd = rndMask.sum()
        angleArray1 = np.angle(unmaskedArray1)
        angleArray2 = np.angle(unmaskedArray2)
        absArray1 = np.abs(unmaskedArray1)
        absArray2 = np.abs(unmaskedArray2)

    # We repeat the phase randomization to average

        correctedFSCum = np.zeros(nShells)
        phaseRndFSCum = np.zeros(nShells)

        for k in range(nIter):
            rndAngleArray1 = angleArray1.copy()
            rndAngleArray2 = angleArray2.copy()

            rndAngleArray1[rndMask] = np.random.rand(nRnd)*2*np.pi
            rndAngleArray2[rndMask] = np.random.rand(nRnd)*2*np.pi

            rndArray1FT = absArray1 * np.exp(1j*rndAngleArray1)
            rndArray2FT = absArray2 * np.exp(1j*rndAngleArray2)

            rndArray1 = np.fft.ifftn(np.fft.ifftshift(rndArray1FT))
            rndArray2 = np.fft.ifftn(np.fft.ifftshift(rndArray2FT))

            maskedRndArray1FT = np.fft.fftshift(np.fft.fftn(rndArray1*mask))
            maskedRndArray2FT = np.fft.fftshift(np.fft.fftn(rndArray2*mask))

            maskedRndCCF = np.real(maskedRndArray1FT * maskedRndArray2FT.conj())
            maskedRndInt1 = np.real(maskedRndArray1FT * maskedRndArray1FT.conj())
            maskedRndInt2 = np.real(maskedRndArray2FT * maskedRndArray2FT.conj())

            maskedRndCCF_1D = np.zeros(nShells)
            maskedRndInt1_1D = np.zeros(nShells)
            maskedRndInt2_1D = np.zeros(nShells)
            for k in range(nShells):
                maskedRndCCF_1D[k] = maskedRndCCF[shellList[k]].sum()
                maskedRndInt1_1D[k] = maskedRndInt1[shellList[k]].sum()
                maskedRndInt2_1D[k] = maskedRndInt2[shellList[k]].sum()

            maskedRndFSC = maskedRndCCF_1D / \
                np.sqrt(maskedRndInt1_1D * maskedRndInt2_1D)

            # Mask-corrected FSC
            correctedFSCTmp = maskedFSC.copy()
            correctedFSCTmp[rndIdx+rndOffset:] = \
                (maskedFSC[rndIdx+rndOffset:] - maskedRndFSC[rndIdx+rndOffset:]) /\
                (1 - maskedRndFSC[rndIdx+rndOffset:])
            correctedFSCum += correctedFSCTmp
            phaseRndFSCum += maskedRndFSC
            correctedFSC = correctedFSCum/nIter
            phaseRndFSC = phaseRndFSCum/nIter

    inv_resolution = np.arange(nShells) / (iSize*samplingRate)
    return inv_resolution, unmaskedFSC, maskedFSC, phaseRndFSC, correctedFSC

def determine_fsc_resolution(inv_resolution, fsc_values, threshold=0.143):

    index = np.where(fsc_values < threshold)[0][0]  #TODO: add try except for IndexError, probably caused by too tight mask
    inv_res_at_threshold = inv_resolution[index - 1] + (inv_resolution[index] - inv_resolution[index - 1]) * (
                threshold - fsc_values[index - 1]) / (fsc_values[index] - fsc_values[index - 1])
    res_at_threshold = 1 / inv_res_at_threshold
    return inv_res_at_threshold, res_at_threshold

def plot_fsc(ax, inv_resolution, fsc_values, inv_res_at_threshold, label):
    ax.plot(inv_resolution, fsc_values, label=label)
    ax.axvline(x=inv_res_at_threshold, color='r', linestyle='-.')
    ticks = inv_resolution[::10]
    ticks[0]=1/999
    ax.set_xticks(ticks)
    ax.set_xticklabels(np.round(1 / ticks, decimals=2))
    ax.set_xlabel('Resolution (Å)')
    ax.set_ylabel('FSC')

def getFSCResolution(array1, array2, mask, samplingRate, resolution_threshold=0.143):
    inv_resolution, unmaskedFSC, maskedFSC, phaseRndFSC, correctedFSC = \
        getMaskCorrectedFSC(array1, array2, mask, samplingRate)
    _, res_unmasked = determine_fsc_resolution(inv_resolution, unmaskedFSC, threshold=resolution_threshold)
    if maskedFSC is not None:
        _, res_masked = determine_fsc_resolution(inv_resolution, maskedFSC, threshold=resolution_threshold)
        _, res_phaseRnd = determine_fsc_resolution(inv_resolution, phaseRndFSC, threshold=resolution_threshold)
        _, res_corrected = determine_fsc_resolution(inv_resolution, correctedFSC, threshold=resolution_threshold)
    else:
        res_masked, res_phaseRnd, res_corrected = [None]*3
    return res_unmasked, res_masked, res_phaseRnd, res_corrected


if __name__ == "__main__":

    parser = argparse.ArgumentParser("FSC calculation")
    parser.add_argument("-i", "--inputMap", required=True, type=str, help="The input map or half map 1")
    parser.add_argument("-r", "--referenceMap", required=True, type=str, help="The reference map or half map 2")
    parser.add_argument("-m", "--mask", required=False, type=str, help="A mask to limit computations. It shoud have soft edges")
    parser.add_argument("-p", "--show_plot", action="store_true",
                        help="Show an interactive FSC resolution plot ")
    parser.add_argument("-o", "--plot_fname", required=False, type=str,
                        help="The filename for the plot to be saved")
    parser.add_argument("-t", "--resolution_threshold", required=False, type=float, default=0.143,
                        help="The thresold for the FSC resolution determination")
    args = parser.parse_args()

    threshold = args.resolution_threshold


    with mrcfile.open(osp.expanduser(args.inputMap)) as f:
        array1 = f.data.copy()
        samplingRate = float(f.voxel_size.x)
    with mrcfile.open(osp.expanduser(args.referenceMap)) as f:
        array2 = f.data.copy()
        assert round(samplingRate,3) == round(float(f.voxel_size.x),3), ("Error, input and reference map have different"
                                                                 " sampling rate")
    if args.mask is not None:
        with mrcfile.open(osp.expanduser(args.mask)) as f:
            mask = f.data.copy()
            assert round(samplingRate,3) == round(float(f.voxel_size.x),3), (f"Error, input map and mask have "
                                                                             f"different sampling rate "
                                                                             f"{samplingRate, f.voxel_size.x}")
    else:
        mask=None
    (inv_resolution, unmaskedFSC, maskedFSC,
     phaseRndFSC, correctedFSC) = getMaskCorrectedFSC(array1, array2, mask, samplingRate)
    inv_res_at_threshold, resolution = determine_fsc_resolution(inv_resolution, unmaskedFSC, threshold=threshold)
    print(f"Resolution {resolution} Å  (Sampling rate {samplingRate})")

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.axhline(y=threshold, color='r', linestyle='--')
    plot_fsc(ax, inv_resolution, unmaskedFSC, inv_res_at_threshold,
             label=f'Unmasked FSC ({np.round(1/inv_res_at_threshold, 2)})')

    if maskedFSC is not None:
        inv_res_at_threshold, maskedResolution = determine_fsc_resolution(inv_resolution, maskedFSC)
        print(f"Masked Resolution {maskedResolution}")
        plot_fsc(ax, inv_resolution, maskedFSC, inv_res_at_threshold,
                 label=f'Masked FSC ({np.round(1/inv_res_at_threshold, 2)})')


    ax.legend()
    plt.show()


"""
-i ~/tmp/ConorData/bound/halfs/docks_simulted_micSim0_half1.mrc -r ~/tmp/ConorData/bound/halfs/docks_simulted_micSim0_half2.mrc
"""