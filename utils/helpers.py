import cv2
import numpy as np
import pywt

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

def visualize_dct_spectrum(img):
    img_float = np.float32(img)
    dct = cv2.dct(img_float)
    # Log transform để dễ nhìn
    spectrum = np.log(np.abs(dct) + 1)
    # Normalize về 0-255
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(spectrum), dct

def visualize_dft_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(magnitude_spectrum), fshift

def visualize_wavelet_spectrum(coeffs):
    # coeffs = (LL, (LH, HL, HH)) hoặc (LL, HL, LH, HH) tùy cấu trúc
    # Chúng ta sẽ ghép các băng tần lại thành 1 ảnh
    if len(coeffs) == 2: # DWT standard format
        LL, (LH, HL, HH) = coeffs
    else: # IWT custom format
        LL, HL, LH, HH = coeffs

    def normalize_band(band):
        return cv2.normalize(np.abs(band), None, 0, 255, cv2.NORM_MINMAX)

    LL_img = normalize_band(LL)
    LH_img = normalize_band(LH)
    HL_img = normalize_band(HL)
    HH_img = normalize_band(HH)

    # Ghép ảnh: 
    # LL | HL
    # ---+---
    # LH | HH
    top = np.hstack((LL_img, HL_img))
    bot = np.hstack((LH_img, HH_img))
    full = np.vstack((top, bot))
    return np.uint8(full)