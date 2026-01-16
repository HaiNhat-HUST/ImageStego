import cv2
import numpy as np
import os
import math
import struct

# --- CẤU HÌNH ---
INPUT_FOLDER = "input_images"  # Thư mục chứa ảnh gốc
OUTPUT_FOLDER = "output_stego"  # Thư mục chứa ảnh sau khi giấu tin
SECRET_MESSAGE = "Day la thong diep bi mat dung de test thuat toan JSteg DCT tren tap du lieu lon."


# --- CLASS XỬ LÝ JSTEG (CORE LOGIC) ---
class JStegDCT:
    def __init__(self):
        # Bảng lượng tử hóa chuẩn (Luminance)
        self.Q_TABLE = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)

    def _text_to_bits(self, text):
        bits = []
        for char in text:
            binval = bin(ord(char))[2:].zfill(8)
            bits.extend([int(b) for b in binval])
        return bits

    def _bits_to_text(self, bits):
        chars = []
        for b in range(len(bits) // 8):
            byte = bits[b * 8:(b + 1) * 8]
            chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
        return "".join(chars)

    def dct_embed(self, img, secret_data):
        h, w = img.shape[:2]
        # Cắt ảnh cho chia hết cho 8
        h = h - (h % 8)
        w = w - (w % 8)
        img = img[:h, :w]

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = y.astype(np.float32)

        bits = self._text_to_bits(secret_data)
        len_bits = [int(b) for b in bin(len(secret_data))[2:].zfill(32)]
        all_bits = len_bits + bits
        bit_idx = 0
        total_bits = len(all_bits)

        dct_blocks = np.zeros_like(y)

        # Duyệt block 8x8
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = y[i:i + 8, j:j + 8]
                dct_block = cv2.dct(block - 128)
                quantized_block = np.round(dct_block / self.Q_TABLE).astype(np.int32)

                if bit_idx < total_bits:
                    for u in range(8):
                        for v in range(8):
                            if bit_idx >= total_bits: break

                            # JSteg logic: Bỏ qua DC(0,0), bỏ qua 0 và 1
                            if u == 0 and v == 0: continue
                            coeff = quantized_block[u, v]
                            if coeff == 0 or coeff == 1: continue

                            bit = all_bits[bit_idx]
                            # Nhúng vào LSB
                            if coeff > 0:
                                if (coeff % 2) != bit:
                                    coeff += 1 if (coeff % 2 == 0) else -1
                            else:
                                if (abs(coeff) % 2) != bit:
                                    coeff -= 1 if (abs(coeff) % 2 == 0) else 1

                            quantized_block[u, v] = coeff
                            bit_idx += 1

                dct_blocks[i:i + 8, j:j + 8] = quantized_block

        # Tái tạo ảnh
        y_stego = np.zeros_like(y)
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                quantized_block = dct_blocks[i:i + 8, j:j + 8]
                dct_block = quantized_block * self.Q_TABLE
                y_stego[i:i + 8, j:j + 8] = cv2.idct(dct_block) + 128

        y_stego = np.clip(y_stego, 0, 255).astype(np.uint8)
        stego_ycrcb = cv2.merge([y_stego, cr, cb])
        return cv2.cvtColor(stego_ycrcb, cv2.COLOR_YCrCb2BGR), dct_blocks

    def dct_extract(self, dct_coefficients):
        extracted_bits = []
        h, w = dct_coefficients.shape

        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = dct_coefficients[i:i + 8, j:j + 8]
                for u in range(8):
                    for v in range(8):
                        if u == 0 and v == 0: continue
                        coeff = block[u, v]
                        if coeff != 0 and coeff != 1:
                            extracted_bits.append(int(abs(coeff) % 2))

        if len(extracted_bits) < 32: return "No data"
        len_bits = extracted_bits[:32]
        try:
            msg_len = int(''.join([str(b) for b in len_bits]), 2)
        except:
            return "Header Error"

        total_msg_bits = 32 + (msg_len * 8)
        if len(extracted_bits) < total_msg_bits: return "Corrupted"

        return self._bits_to_text(extracted_bits[32:total_msg_bits])


# --- CÁC HÀM TIỆN ÍCH ---
def calculate_psnr(img1, img2):
    # Resize img2 về kích thước img1 nếu bị cắt (do crop 8x8)
    h, w = img1.shape[:2]
    img2 = img2[:h, :w]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))


def safe_imread(path):
    # Đọc ảnh an toàn trên Windows (tránh lỗi đường dẫn tiếng Việt/Ký tự lạ)
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    except:
        return None


# --- MAIN ---
if __name__ == "__main__":
    # 1. Tạo thư mục nếu chưa có
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"[THÔNG BÁO] Đã tạo thư mục '{INPUT_FOLDER}'. Hãy copy ảnh vào đây rồi chạy lại!")
        exit()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 2. Lấy danh sách ảnh
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"[LỖI] Thư mục '{INPUT_FOLDER}' đang trống. Hãy thêm ảnh vào!")
        exit()

    stego_solver = JStegDCT()

    # 3. In tiêu đề báo cáo
    print(f"\n{'TÊN FILE':<25} | {'KÍCH THƯỚC':<15} | {'PSNR (dB)':<10} | {'TRẠNG THÁI':<10}")
    print("-" * 70)

    # 4. Xử lý từng file
    for file_name in files:
        input_path = os.path.join(INPUT_FOLDER, file_name)
        # Lưu output dạng PNG để tránh nén JPEG làm hỏng tin
        output_name = os.path.splitext(file_name)[0] + "_stego.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        # Đọc ảnh
        original_img = safe_imread(input_path)
        if original_img is None:
            print(f"{file_name:<25} | {'LỖI ĐỌC FILE':<15} | {'N/A':<10} | {'ERROR'}")
            continue

        try:
            # Nhúng tin
            stego_img, coeffs = stego_solver.dct_embed(original_img, SECRET_MESSAGE)

            # Lưu kết quả
            cv2.imwrite(output_path, stego_img)

            # Tính PSNR (So với ảnh gốc đã crop theo kích thước stego)
            h, w = stego_img.shape[:2]
            cropped_orig = original_img[:h, :w]
            psnr_val = calculate_psnr(cropped_orig, stego_img)

            # Kiểm tra giải mã
            decoded_msg = stego_solver.dct_extract(coeffs)
            status = "SUCCESS" if decoded_msg == SECRET_MESSAGE else "FAIL"

            # In kết quả
            dims = f"{w}x{h}"
            print(f"{file_name:<25} | {dims:<15} | {psnr_val:<10.2f} | {status}")

        except Exception as e:
            print(f"{file_name:<25} | {'ERROR':<15} | {'N/A':<10} | {str(e)}")

    print("-" * 70)
    print(f"\n[HOÀN TẤT] Ảnh kết quả đã lưu tại thư mục: {os.path.abspath(OUTPUT_FOLDER)}")