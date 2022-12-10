import skimage.io as io
from skimage.filters import gaussian
from skimage import metrics
from deconv_admm_tv import *
from deconv_admm_dncnn import *
import os
from pathlib import Path
import pandas as pd
from glob import glob


def fspecial_gaussian_2d(size, s):
    kernel = np.zeros(tuple(size))
    kernel[size[0]//2, size[1]//2] = 1
    kernel = gaussian(kernel, s)
    return kernel/np.sum(kernel)

data_dir = Path(__file__).parent / 'processed'
result_folder = data_dir / 'result'
model_name = 'admm_dncnn'
model_file = 'admm_dncnn.csv'
result_file = result_folder / model_file

if glob(result_file.as_posix()):
    print("result file exist")
    resultdf = pd.read_csv(result_file)
else:
    print("result file DNE, create new file!")
    column_names = ['model', 'noise', 'type', 'PSNR', 'MSE']
    resultdf = pd.DataFrame(columns=column_names)

for sigma in [10, 50]:
    noise_type = 'gau' + str(sigma)
    noise_folder = data_dir / 'test/Low' / noise_type
    images = sorted(list(noise_folder.glob('*.png')))
    # blur kernel
    c = fspecial_gaussian_2d((15, 15), sigma / 255)
    for image in images:
        file_name = os.path.basename(image)
        img = io.imread(image).astype(np.float64) / 255
        ori_name = data_dir / 'test/Low/original' / file_name
        ori_img = io.imread(f'{ori_name}').astype(float) / 255

        # Blur kernel
        cFT = psf2otf(c, (img.shape[0], img.shape[1]))
        Afun = lambda x: np.real(ifft2(fft2(x) * cFT))

        # simulated measurements
        b = np.zeros(np.shape(img))
        b[:, :] = Afun(img[:, :]) + 10 / 255 / 15 * np.random.randn(img.shape[0], img.shape[1])

        # ADMM parameters for TV prior
        num_iters = 75
        rho = 10
        lam = 0.03

        # run ADMM+TV solver
        x_admm_tv = np.zeros(np.shape(b))
        x_admm_tv[:, :] = deconv_admm_tv(b[:, :], c, lam, rho, num_iters)
        x_admm_tv = np.clip(x_admm_tv, 0.0, 1.0)
        PSNR_n = round(metrics.peak_signal_noise_ratio(img, x_admm_tv), 1)
        MSE_n = metrics.mean_squared_error(img, x_admm_tv)
        PSNR_o = round(metrics.peak_signal_noise_ratio(img, ori_img), 1)
        MSE_o = metrics.mean_squared_error(img, ori_img)
        new_result_df = pd.DataFrame([[model_name, noise_type, 'original', PSNR_o, MSE_o]],
                                     columns=['model', 'noise', 'type', 'PSNR', 'MSE'])
        resultdf = resultdf.append(new_result_df, ignore_index=True)
        new_result_df = pd.DataFrame([[model_name, noise_type, 'denoised', PSNR_n, MSE_n]],
                                     columns=['model', 'noise', 'type', 'PSNR', 'MSE'])
        resultdf = resultdf.append(new_result_df, ignore_index=True)
        resultdf.to_csv(result_file, index=False, header=True)
        io.imsave(result_folder / 'admm_tv' / noise_type / file_name, (np.clip(x_admm_tv, 0, 1) * 255).astype(np.uint8))
        io.imsave(result_folder / 'noise' / noise_type / file_name, (np.clip(b, 0, 1) * 255).astype(np.uint8))

        # # ADMM parameters for DnCNN prior
        # num_iters = 75
        # lam = 0.007
        # rho = 0.45
        #
        # # run ADMM+DnCNN solver
        # x_admm_dncnn = np.zeros(np.shape(b))
        # for it in range(3):
        #     x_admm_dncnn[:, :] = deconv_admm_dncnn(b[:, :], c, lam, rho, num_iters)
        # x_admm_dncnn = np.clip(x_admm_dncnn, 0.0, 1.0)
        # PSNR_ADMM_dncnn = round(metrics.peak_signal_noise_ratio(img, x_admm_dncnn), 1)
        # MSE_ADMM_dncnn = round(metrics.mean_squared_error(img, x_admm_dncnn), 1)
        # PSNR_o = round(metrics.peak_signal_noise_ratio(img, ori_img), 1)
        # MSE_o = round(metrics.mean_squared_error(img, ori_img), 1)
        # new_result_df = pd.DataFrame([[model_name, noise_type, 'original', PSNR_o, MSE_o]],
        #                              columns=['model', 'noise', 'type', 'PSNR', 'MSE'])
        # resultdf = resultdf.append(new_result_df, ignore_index=True)
        # new_result_df = pd.DataFrame([[model_name, noise_type, 'denoised', PSNR_n, MSE_n]],
        #                              columns=['model', 'noise', 'type', 'PSNR', 'MSE'])
        # resultdf = resultdf.append(new_result_df, ignore_index=True)
        # resultdf.to_csv(result_file, index=False, header=True)
        # io.imsave(result_folder / 'admm_dncnn' / noise_type / file_name, (np.clip(x_admm_dncnn, 0, 1) * 255).astype(np.uint8))