import streamlit as st
import cv2  
import numpy as np
import pywt
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from utils.text_bin import text_to_bin, bin_to_text
from algorithms.dwt import apply_dwt, apply_idwt  

def embed_bits_in_subband(subband, binary_messages):

    flat_subband = subband.flatten()
    modified_subband = flat_subband.copy()

    msg_len = len(binary_messages)

    bin_len = format(msg_len, '032b')
    full_message = bin_len + binary_messages

    if (len(full_message) > len(flat_subband)):
        st.error(f"Message too long to embed in the selected subband. Need {len(full_message)} bits, but only {len(flat_subband)} available.")
        return subband, False
    
    for i in range(len(full_message)):
        val = int(modified_subband[i])
        bit = int(full_message[i])

        new_val = (val & ~1) | bit 

        modified_subband[i] = float(new_val)

    return modified_subband.reshape(subband.shape), True
    
def extract_bits_from_subband(subband):

    flat_subband = subband.flatten()
    extracted_bits = ""

    for i in range(32):
        val = int(flat_subband[i])
        extracted_bits += str(val & 1)

    try:

        msg_len = int(extracted_bits, 2)
    except:
        return "Failed to extract message length."

    message_bits = ""

    for i in range(32,32 + msg_len):
        if i >= len (flat_subband): break
        val = int(flat_subband[i])
        message_bits += str(val & 1)

    return bin_to_text(message_bits)

def plot_frequency_domain(coeffs, title="Frequency Domain with DWT"):

    LL, (LH, HL, HH) = coeffs
    
    def normalize(arr):
        arr = np.abs(arr)
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-5) * 255
        return arr.astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    ax = axes.ravel()
    
    ax[0].imshow(normalize(LL), cmap='gray')
    ax[0].set_title("LL (Low-Freq)\nApproximation")
    
    ax[1].imshow(normalize(LH), cmap='gray')
    ax[1].set_title("LH (Horizontal)")
    
    ax[2].imshow(normalize(HL), cmap='gray')
    ax[2].set_title("HL (Vertical)")
    
    ax[3].imshow(normalize(HH), cmap='gray')
    ax[3].set_title("HH (Diagonal)\nSecret Data Here")
    
    for a in ax: a.axis('off')
    plt.tight_layout()
    return fig

st.title("Digital Image Steganography in Transform Domain using DWT")

with st.sidebar:
    st.header("Algorithm Configuration")
    image_mode = st.radio("Image type", ["Grayscale", "Color (RGB)"], index=1)

    color_space = None
    if image_mode == "Color (RGB)":
        color_space = st.selectbox("Color Space for Embedding", ["RGB (Embed in Blue channel)", "YCbCr (Embed in Cb channel)"], help="YCbCr may provide better imperceptibility because human vision is less sensitive to chrominance changes.")

    secret_msg = st.text_area("Secret Message to Embed", "Hello, this is a secret message!")
    st.info("Note: Message will be embed in the high-frequency HH subband of the selected color channel.")

uploaded_file = st.file_uploader("Upload an Image (PNG, JPG, JPEG, TIFF)", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

if uploaded_file and secret_msg:

    image_pil = Image.open(uploaded_file)
    image_np = np.array(image_pil)
    
    st.subheader("1. Analysis and Embedding Process")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_pil, caption="Original Image", use_container_width=True)

    stego_image_np = None
    coeffs_original = None
    coeffs_stego = None
    extracted_text = ""
    
    if image_mode == "Grayscale":

        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
            
        # DWT Transform
        coeffs = apply_dwt(gray)
        coeffs_original = coeffs 
        LL, (LH, HL, HH) = coeffs
        
        # Embedding
        bin_msg = text_to_bin(secret_msg)
        HH_new, success = embed_bits_in_subband(HH, bin_msg)
        
        if success:
            coeffs_mod = (LL, (LH, HL, HH_new))
            coeffs_stego = coeffs_mod
            
            # IDWT Reconstruction
            stego_float = apply_idwt(coeffs_mod)
            stego_image_np = np.clip(stego_float, 0, 255).astype(np.uint8)
            
            # Extraction simulation
            extracted_text = extract_bits_from_subband(HH_new)

    else: 
        if "YCbCr" in color_space:
            # Chuyển đổi RGB -> YCbCr
            img_ycbcr = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
            Y, Cr, Cb = cv2.split(img_ycbcr)
            
            coeffs = apply_dwt(Cb)
            coeffs_original = coeffs
            LL, (LH, HL, HH) = coeffs
            
            bin_msg = text_to_bin(secret_msg)
            HH_new, success = embed_bits_in_subband(HH, bin_msg)
            
            if success:
                coeffs_mod = (LL, (LH, HL, HH_new))
                coeffs_stego = coeffs_mod
                
                Cb_new_float = apply_idwt(coeffs_mod)
                Cb_new = np.clip(Cb_new_float, 0, 255).astype(np.uint8)
                
                # Merge lại
                stego_ycbcr = cv2.merge([Y, Cr, Cb_new])
                stego_image_np = cv2.cvtColor(stego_ycbcr, cv2.COLOR_YCrCb2RGB)
                extracted_text = extract_bits_from_subband(HH_new)
                
        else:
            R, G, B = cv2.split(image_np)
            
            coeffs = apply_dwt(B)
            coeffs_original = coeffs
            LL, (LH, HL, HH) = coeffs
            
            bin_msg = text_to_bin(secret_msg)
            HH_new, success = embed_bits_in_subband(HH, bin_msg)
            
            if success:
                coeffs_mod = (LL, (LH, HL, HH_new))
                coeffs_stego = coeffs_mod
                
                B_new_float = apply_idwt(coeffs_mod)
                B_new = np.clip(B_new_float, 0, 255).astype(np.uint8)
                
                stego_image_np = cv2.merge([R, G, B_new])
                extracted_text = extract_bits_from_subband(HH_new)

    with col2:
        if stego_image_np is not None:
            st.image(stego_image_np, caption="Stego Image", use_container_width=True)
            
            # Tính PSNR
            mse = np.mean((image_np.astype(float) - stego_image_np.astype(float)) ** 2)
            psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else 100
            st.metric("Image quality (PSNR)", f"{psnr:.2f} dB", help="Value >30dB is considered good.")

    st.subheader("2. Compare Frequency Domain Before and After Embedding")
    
    if coeffs_original and coeffs_stego:
        freq_col1, freq_col2 = st.columns(2)
        with freq_col1:
            st.write("Bẻoefore Embedding")
            fig1 = plot_frequency_domain(coeffs_original)
            st.pyplot(fig1)
        
        with freq_col2:
            st.write("After Embedding")
            fig2 = plot_frequency_domain(coeffs_stego)
            st.pyplot(fig2)

    st.subheader("3. Extracted Secret Message")
    st.success(f"Extract message from stego image **{extracted_text}**")
    
else:
    st.info("Upload an image and enter a secret message to embed using DWT-based steganography.")