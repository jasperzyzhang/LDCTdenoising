import numpy as np
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf
from tqdm import tqdm

def deconv_admm_tv(b, c, lam, rho, num_iters, anisotropic_tv=False):

    # Blur kernel
    cFT = psf2otf(c, b.shape)
    cTFT = np.conj(cFT)

    # First differences
    dx = np.array([[-1., 1.]])
    dy = np.array([[-1.], [1.]])
    dxFT = psf2otf(dx, b.shape)
    dyFT = psf2otf(dy, b.shape)
    dxTFT = np.conj(dxFT)
    dyTFT = np.conj(dyFT)
    dxyFT = np.stack((dxFT, dyFT), axis=0)

    # Fourier transform of b 
    bFT = fft2(b)

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros((2, *b.shape))
    u = np.zeros((2, *b.shape))

    grad_fn = lambda x: ifft2(dxyFT * fft2(np.stack((x, x), axis=0)))

    # precompute the denominator for the x-update
    denom = cTFT * cFT + rho * (dxTFT * dxFT + dyTFT * dyFT)

    for it in tqdm(range(num_iters)):
        v = z - u
        v1_f = fft2(v[0, :, :])
        v2_f = fft2(v[1, :, :])
        x_f = (cTFT * bFT + rho * (dxTFT * v1_f + dyTFT * v2_f)) / denom
        x = ifft2(x_f)

        # z update - soft shrinkage
        kappa = lam / rho
        v = grad_fn(x) + u

        # proximal operator of anisotropic TV term
        if anisotropic_tv:
            z = np.maximum(1 - kappa/np.abs(v), 0) * v

        # proximal operator of isotropic TV term
        else:
            vnorm = np.sqrt( v[0,:,:]**2 + v[1,:,:]**2 )
            z[0,:,:] = np.maximum(1 - kappa/vnorm,0) * v[0,:,:]
            z[1,:,:] = np.maximum(1 - kappa/vnorm,0) * v[1,:,:]

        # u-update
        u = u + grad_fn(x) - z

    return x

def deconv_admm_tv_poi(b, cFT, lam, rho, num_iters, anisotropic_tv=False):

    # Blur kernel
    cTFT = np.conj(cFT)

    # First differences
    dx = np.array([[-1., 1.]])
    dy = np.array([[-1.], [1.]])
    dxFT = psf2otf(dx, b.shape)
    dyFT = psf2otf(dy, b.shape)
    dxTFT = np.conj(dxFT)
    dyTFT = np.conj(dyFT)
    dxyFT = np.stack((dxFT, dyFT), axis=0)

    # Fourier transform of b
    bFT = fft2(b)

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros((2, *b.shape))
    u = np.zeros((2, *b.shape))
    grad_fn = lambda x: ifft2(dxyFT * fft2(np.stack((x, x), axis=0)))

    # precompute the denominator for the x-update
    denom = cTFT * cFT + rho * (dxTFT * dxFT + dyTFT * dyFT)

    for it in tqdm(range(num_iters)):
        v = z - u
        v1_f = fft2(v[0, :, :])
        v2_f = fft2(v[1, :, :])
        x_f = (cTFT * bFT + rho * (dxTFT * v1_f + dyTFT * v2_f)) / denom
        x = ifft2(x_f)

        # z update - soft shrinkage
        kappa = lam / rho
        v = grad_fn(x) + u

        # proximal operator of anisotropic TV term
        if anisotropic_tv:
            z = np.maximum(1 - kappa/np.abs(v), 0) * v

        # proximal operator of isotropic TV term
        else:
            vnorm = np.sqrt( v[0,:,:]**2 + v[1,:,:]**2 )
            z[0,:,:] = np.maximum(1 - kappa/vnorm,0) * v[0,:,:]
            z[1,:,:] = np.maximum(1 - kappa/vnorm,0) * v[1,:,:]

        # u-update
        u = u + grad_fn(x) - z

    return x
