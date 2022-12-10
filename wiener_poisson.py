import skimage.io as io
from skimage.filters import gaussian
from skimage import metrics
import numpy as np
import os
from pathlib import Path
import pandas as pd
from glob import glob
from numpy.fft import fft2, ifft2

data_dir = Path(__file__).parent / 'processed'
result_folder = data_dir / 'result'
model_name = 'wiener'
model_file = 'wiener.csv'
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
        blur = Afun(ori_img)
        b[:, :] = blur + 10 / 255 / 15 * np.random.randn(img.shape[0], img.shape[1])

        unfilt_f = fft2(b)
        snr = blur.mean() / (10 / 255 / 15)
        wiener = 1 / cFT * np.real(cFT) ** 2 / (np.real(cFT) ** 2 + 1 / snr)
        filt_f = unfilt_f * wiener
        x_wiener = np.real(ifft2(filt_f))

        x_wiener = np.clip(x_wiener, 0.0, 1.0)
        PSNR_n = round(metrics.peak_signal_noise_ratio(img, x_wiener), 1)
        MSE_n = metrics.mean_squared_error(img, x_wiener)
        PSNR_o = round(metrics.peak_signal_noise_ratio(img, ori_img), 1)
        MSE_o = metrics.mean_squared_error(img, ori_img)
        new_result_df = pd.DataFrame([[model_name, noise_type, 'original', PSNR_o, MSE_o]],
                                     columns=['model', 'noise', 'type', 'PSNR', 'MSE'])
        resultdf = resultdf.append(new_result_df, ignore_index=True)
        new_result_df = pd.DataFrame([[model_name, noise_type, 'denoised', PSNR_n, MSE_n]],
                                     columns=['model', 'noise', 'type', 'PSNR', 'MSE'])
        resultdf = resultdf.append(new_result_df, ignore_index=True)
        resultdf.to_csv(result_file, index=False, header=True)
        io.imsave(result_folder / 'wiener' / noise_type / file_name, (np.clip(x_wiener, 0, 1) * 255).astype(np.uint8))