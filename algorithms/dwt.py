import pywt

def apply_dwt(channel):
    coeffs = pywt.dwt2(channel, 'haar')
    return coeffs

def apply_idwt(coeffs):
    return pywt.idwt2(coeffs, 'haar')