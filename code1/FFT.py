import cv2
import numpy as np
from math import sqrt

import torch


# def amplitude_spectrum_mix2(img1, img2, alpha):
#     """Apply Amplitude Spectrum Mix to two images and return mixed images."""
#     # Ensure img1 and img2 are numpy arrays with shape [H, W, C]
#     img1_fft = np.fft.fft2(img1, axes=(0, 1))
#     img2_fft = np.fft.fft2(img2, axes=(0, 1))
#     img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
#     img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)
#
#     # Mixing the amplitudes
#     mixed_abs = alpha * img2_abs + (1 - alpha) * img1_abs
#
#     # Reconstruct image from mixed amplitude and original phase
#     mixed_fft = mixed_abs * np.exp(1j * img1_pha)
#     mixed_img = np.fft.ifft2(mixed_fft, axes=(0, 1)).real
#     mixed_img = np.clip(mixed_img, 0, 255)  # Ensure pixel values are valid
#     return mixed_img.astype(np.uint8)
def amplitude_spectrum_mix(img1, img2, alpha, ratio=1.0):   #img_src, img_random
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)
    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)

    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img_src_random = img1_abs * (np.e ** (1j * img1_pha))
    img_src_random = np.real(np.fft.ifft2(img_src_random, axes=(0, 1)))
    img_src_random = np.uint8(np.clip(img_src_random, 0, 255))
    return img_src_random


# def amplitude_spectrum_mix(img1, img2, alpha):
#     """Apply Amplitude Spectrum Mix to two images and return mixed images, handling batches."""
#     assert img1.shape == img2.shape
#     # Assume img1 and img2 are of shape [batch_size, channels, height, width]
#     print('test7')
#     print(img1.shape)
#     print(img2.shape)
#
#
#     batch_size, channels, height, width = img1.shape
#
#     # You might need to adjust how you handle batches here
#     result_imgs = []
#     for b in range(batch_size):
#         img1_b = img1[b].permute(1, 2, 0).cpu().numpy()  # Converting to [height, width, channels]
#         img2_b = img2[b].permute(1, 2, 0).cpu().numpy()  # Converting to [height, width, channels]
#
#         img1_fft = np.fft.fft2(img1_b, axes=(0, 1))
#         img2_fft = np.fft.fft2(img2_b, axes=(0, 1))
#         img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
#         img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)
#
#         # Mixing the amplitudes
#         mixed_abs = alpha * img2_abs + (1 - alpha) * img1_abs
#
#         # Reconstruct image from mixed amplitude and original phase
#         mixed_fft = mixed_abs * np.exp(1j * img1_pha)
#         mixed_img = np.fft.ifft2(mixed_fft, axes=(0, 1)).real
#         mixed_img = np.clip(mixed_img, 0, 255)
#
#         # Convert back to tensor and append to results
#         mixed_img_tensor = torch.from_numpy(mixed_img).to(img1.device).float()
#         mixed_img_tensor = mixed_img_tensor.permute(2, 0, 1)  # Back to [channels, height, width]
#         result_imgs.append(mixed_img_tensor.unsqueeze(0))
#
#     return torch.cat(result_imgs, dim=0)  # Concatenate along the batch dimension
