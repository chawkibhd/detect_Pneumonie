import os
import csv
import numpy as np
from scipy import ndimage as ndi
from skimage import io, img_as_float
from skimage.filters import gabor_kernel
from skimage.transform import resize
from scipy.fftpack import dct as scipy_dct
import cv2
from skimage.feature import hog

class ImageFeatureExtractor:
    def __init__(self, base_dir, csv_file_path, target_size=(128, 128)):
        self.base_dir = base_dir
        self.csv_file_path = csv_file_path
        self.target_size = target_size
        self.kernels = [
            np.real(gabor_kernel(frequency, theta=theta / 4.0 * np.pi, sigma_x=sigma, sigma_y=sigma))
            for theta in range(4)
            for sigma in (1, 3)
            for frequency in (0.05, 0.25)
        ]

    def load_and_preprocess_image(self, image_path):
        image = img_as_float(io.imread(image_path, as_gray=True))
        resized_image = resize(image, self.target_size, anti_aliasing=True)
        return resized_image

    def compute_gabor(self, image):
        feats = np.array([(ndi.convolve(image, kernel, mode='wrap').mean(),
                           ndi.convolve(image, kernel, mode='wrap').var()) for kernel in self.kernels])
        return feats.flatten()

    def compute_dct(self, image, num_coeffs=100):
        dct_flat = scipy_dct(scipy_dct(image, axis=0, norm='ortho'), axis=1, norm='ortho').flatten()
        return dct_flat[:num_coeffs]

    def compute_ft(self, image, num_coeffs=100):
        fft_magnitude = np.abs(np.fft.fft2(image)).flatten()
        return np.sort(fft_magnitude)[:num_coeffs]

    def compute_phog(self, image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                     pyramid_levels=3, num_coeffs=100):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pyramid_features = []
        for level in range(pyramid_levels):
            scaled_image = cv2.resize(image, (0, 0), fx=1 / (2 ** level), fy=1 / (2 ** level))
            hog_features = hog(scaled_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                               cells_per_block=cells_per_block, block_norm='L2-Hys', visualize=False, feature_vector=True)
            pyramid_features.extend(hog_features)
        return np.array(pyramid_features)[:num_coeffs]

    def extract_features(self, sub_dirs):
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Numéro d'image", "Gabor", "DCT", "Fourier", "PHOG", "Étiquette"])

            for sub_dir in sub_dirs:
                folder_path = os.path.join(self.base_dir, sub_dir)
                label = 0 if sub_dir == "NORMAL" else 1

                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(folder_path, filename)

                            image = self.load_and_preprocess_image(image_path)

                            gabor_feats = self.compute_gabor(image)
                            dct_feats = self.compute_dct(image, num_coeffs=20)
                            ft_feats = self.compute_ft(image, num_coeffs=20)
                            phog_feats = self.compute_phog(image, num_coeffs=20)

                            gabor_mean = np.mean(gabor_feats)
                            dct_mean = np.mean(dct_feats)
                            ft_mean = np.mean(ft_feats)
                            phog_mean = np.mean(phog_feats)

                            numero_image = filename.split('_')[-1].split('.')[0]

                            writer.writerow([numero_image, gabor_mean, dct_mean, ft_mean, phog_mean, label])
                            print(f"Processed {filename}")

        print(f"Image features saved to '{self.csv_file_path}'.")