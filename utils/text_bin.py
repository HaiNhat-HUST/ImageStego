def text_to_bin(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bin_to_text(binary):
    try:
        chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
        return ''.join(chr(int(b, 2)) for b in chars if len(b) == 8)
    except:
        return ""