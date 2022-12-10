import numpy as np
import skimage.io as io
from skimage.util import random_noise
import os
from pathlib import Path

data_dir = Path(__file__).parent / 'processed'
low_folder = data_dir / 'test/Low'
noise_folder = data_dir / 'test/Noise'
images = sorted(list(low_folder.glob('*.png')))

for img in images:
    file_name = os.path.basename(img)
    i = io.imread(img).astype(np.float64) / 255
    # mask = np.random.poisson(i)
    # noisy = i + mask
    noisy = random_noise(i, mode="poisson")
    io.imsave(noise_folder / file_name, (np.clip(noisy, 0, 1) * 255).astype(np.uint8))