import skimage.io as io
from skimage import metrics
from deconv_admm_tv import *
from deconv_admm_dncnn import *
import os
from pathlib import Path
import pandas as pd
from glob import glob

data_dir = Path(__file__).parent / 'processed'
result_folder = data_dir / 'result'
model_name = 'admm_tv'
model_file = 'admm_tv.csv'
result_file = result_folder / model_file

if glob(result_file.as_posix()):
    print("result file exist")
    resultdf = pd.read_csv(result_file)
else:
    print("result file DNE, create new file!")
    column_names = ['model', 'noise', 'type', 'PSNR', 'MSE']
    resultdf = pd.DataFrame(columns=column_names)

for noise_type in ['poi50', 'poiimg']:
    noise_folder = data_dir / 'test/Low' / noise_type
    images = sorted(list(noise_folder.glob('*.png')))
    for image in images:
        file_name = os.path.basename(image)
        img = io.imread(image).astype(np.float64) / 255
        ori_name = data_dir / 'test/Low/original' / file_name
        ori_img = io.imread(f'{ori_name}').astype(float) / 255

        if noise_type == 'poi50':
            poi50 = np.random.poisson(lam=50, size=ori_img.shape)
            poi_img = np.uint8((np.clip(ori_img * 255 + poi50, 0, 255))) / 255
        else:
            vals = len(np.unique(ori_img))
            vals = 2 ** np.ceil(np.log2(vals))
            poi_img = np.uint8(np.clip(np.random.poisson(ori_img * 255 * vals) / float(vals), 0, 255)) / 255

        # Blur kernel
        cFT = fft2(poi_img) / fft2(ori_img)
        Afun = lambda x: np.real(ifft2(fft2(x) * cFT))

        # simulated measurements
        b = np.zeros(np.shape(img))
        b[:, :] = Afun(ori_img) + 10 / 255 / 15 * np.random.randn(img.shape[0], img.shape[1])
        io.imsave(result_folder / 'noise' / noise_type / file_name, (np.clip(b, 0, 1) * 255).astype(np.uint8))

        # ADMM parameters for TV prior
        num_iters = 75
        rho = 3
        lam = 0.01

        # run ADMM+TV solver
        x_admm_tv = np.zeros(np.shape(b))
        x_admm_tv[:, :] = deconv_admm_tv_poi(b[:, :], cFT, lam, rho, num_iters)
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

        # # ADMM parameters for DnCNN prior
        # num_iters = 75
        # lam = 0.045
        # rho = 0.5
        #
        # # run ADMM+DnCNN solver
        # x_admm_dncnn = np.zeros(np.shape(b))
        # for it in range(3):
        #     x_admm_dncnn[:, :] = deconv_admm_dncnn_poi(b[:, :], cFT, lam, rho, num_iters)
        # x_admm_dncnn = np.clip(x_admm_dncnn, 0.0, 1.0)
        # PSNR_n = round(metrics.peak_signal_noise_ratio(img, x_admm_dncnn), 1)
        # MSE_n = round(metrics.mean_squared_error(img, x_admm_dncnn), 1)
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