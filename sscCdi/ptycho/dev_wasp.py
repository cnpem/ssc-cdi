import numpy as np
import cupy as cp
from numpy.fft import fftshift, fft2, ifft2
import matplotlib.pyplot as plt

def WASP(expt, recon, probe):
    """
    An implementation of the Weighted Average of Sequential Projections
    ptychographic algorithm.
    
    *** INPUTS ***
    
    expt: a dictionary containing the experimental parameters and data,
    with the following keys:
        - dps: the recorded diffraction intensities, held in an M x N x D array
        - positions: a dictionary with 'x' and 'y' keys for the scan grid positions
        - wavelength: the beam wavelength in meters
        - cameraPixelPitch: the pixel spacing of the detector
        - cameraLength: the geometric magnification at the front face of the sample
    
    recon: a dictionary containing the reconstruction parameters, with the
    following keys:
        - iters: the number of iterations to carry out
        - gpu: a flag indicating whether to transfer processing to a CUDA-enabled GPU
        - alpha: the object step size parameter (~2)
        - beta: the probe step size parameter (~1)
        - upLimit: the maximum amplitude of the object
    
    probe: an initial model of the probe wavefront
    
    *** OUTPUTS ***
    
    obj: the reconstructed object
    
    probe: the reconstructed probe
    """

    # Pre-processing steps

    # Shift the positions to positive values
    expt['positions']['x'] -= np.min(expt['positions']['x'])
    expt['positions']['y'] -= np.min(expt['positions']['y'])

    # Compute pixel pitch in the sample plane
    M, N, _ = expt['dps'].shape
    dx = expt['wavelength'] * expt['cameraLength'] / (np.array([M, N]) * expt['cameraPixelPitch'])

    # Convert positions to top left (tl) and bottom right (br) pixel locations for each sample position
    tlY = np.round(expt['positions']['y'] / dx[0]).astype(int)
    tlX = np.round(expt['positions']['x'] / dx[1]).astype(int)
    brY = tlY + M - 1
    brX = tlX + N - 1

    # Variable initializations

    # Initialize the "object" as free-space
    obj = np.ones((np.max(brY), np.max(brX)), dtype=np.complex64)

    # Find a suitable probe power from the brightest diffraction pattern
    brightest_idx = np.argmax(np.sum(expt['dps'], axis=(0, 1)))
    probePower = np.sum(expt['dps'][:, :, brightest_idx])

    # Correct the initial probe's power
    probe *= np.sqrt(probePower / (np.prod(probe.shape) * np.sum(np.abs(probe)**2)))

    # Pre-square-root and pre-fftshift the diffraction patterns (for speed)
    expt['dps'] = fftshift(fftshift(np.sqrt(expt['dps']), axes=0), axes=1)

    # Zero-division constant
    c = 1e-10

    # Simple display
    fig, ax = plt.subplots()
    imH = ax.imshow(np.angle(obj), cmap='gray')
    ax.set_aspect('equal')

    # Load variables onto GPU if required
    if recon['gpu']:
        obj = cp.array(obj, dtype=cp.float32)
        probe = cp.array(probe, dtype=cp.float32)
        expt['dps'] = cp.array(expt['dps'], dtype=cp.float32)

    for k in range(recon['iters']):
        # Initialize numerator and denominator sums
        numP = 0 * probe
        denP = 0 * probe
        numO = 0 * obj
        denO = 0 * obj

        # Randomize the diffraction pattern order for sequential projections
        shuffleOrder = np.random.permutation(expt['dps'].shape[2])

        for j in shuffleOrder:
            # Update exit wave to conform with diffraction data
            objBox = obj[tlY[j]:brY[j], tlX[j]:brX[j]]
            currentEW = probe * objBox
            revisedEW = ifft2(expt['dps'][:, :, j] * np.sign(fft2(currentEW)))

            # Sequential projection update of object and probe
            obj[tlY[j]:brY[j], tlX[j]:brX[j]] = objBox + \
                np.conj(probe) * (revisedEW - currentEW) / (np.abs(probe)**2 + recon['alpha'] * np.mean(np.abs(probe)**2))

            probe += np.conj(objBox) * (revisedEW - currentEW) / (np.abs(objBox)**2 + recon['beta'])

            # Update numerator and denominator sums
            numO[tlY[j]:brY[j], tlX[j]:brX[j]] += np.conj(probe) * revisedEW
            denO[tlY[j]:brY[j], tlX[j]:brX[j]] += np.abs(probe)**2
            numP += np.conj(objBox) * revisedEW
            denP += np.abs(objBox)**2

        # Weighted average update of object and probe
        obj = numO / (denO + c)
        probe = numP / (denP + c)

        # Apply additional constraints:

        # Limit hot pixels
        tooHigh = np.abs(obj) > recon['upLimit']
        obj[tooHigh] = recon['upLimit'] * np.sign(obj[tooHigh])

        # Recenter probe/object using probe intensity center of mass
        absP2 = np.abs(probe)**2
        cp = np.fix([M, N] / 2 - [M, N] * [np.mean(np.cumsum(np.sum(absP2, axis=1))),
                                           np.mean(np.cumsum(np.sum(absP2, axis=0)))] / np.sum(absP2) + 1).astype(int)

        if any(cp):
            probe = np.roll(probe, -cp, axis=(0, 1))
            obj = np.roll(obj, -cp, axis=(0, 1))

        # Update display
        imH.set_data(np.angle(obj.get() if recon['gpu'] else obj))
        plt.draw()
        plt.pause(0.001)

    # Format probe and obj for return
    probe = probe.get() if recon['gpu'] else probe
    obj = obj.get() if recon['gpu'] else obj

    return obj, probe
