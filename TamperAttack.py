# code executed by Faizal Nujumudeen
# Presidency University, Bengaluru

import os
import cv2
import numpy as np
import hashlib
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# Utility Functions
# =============================

def sha256(data: bytes):
    return hashlib.sha256(data).digest()

def key_stream(hash_bytes, length):
    return np.frombuffer(hash_bytes, dtype=np.uint8)[:length]

# =============================
# Encryption / Decryption
# =============================

def encrypt_image(image, key=b'secret_key', iv=b'init_vector'):
    flat = image.flatten()
    encrypted = np.zeros_like(flat)

    prev_hash = sha256(key + iv)

    for i in range(len(flat)):
        h_i = sha256(prev_hash + key + i.to_bytes(4, 'big'))
        ks = key_stream(h_i, 1)[0]
        encrypted[i] = flat[i] ^ ks
        #prev_hash = h_i
        prev_hash = sha256(h_i + bytes([encrypted[i]]))

    return encrypted.reshape(image.shape)

def decrypt_image(image, key=b'key', iv=b'iv'):
    image = image.astype(np.uint8)
    flat = image.reshape(-1)

    dec = np.zeros_like(flat)

    prev_hash = hashlib.sha256(key + iv).digest()

    for i in range(len(flat)):
        h = hashlib.sha256(prev_hash + key + i.to_bytes(4,'big')).digest()
        ks = h[0]
        dec[i] = flat[i] ^ ks
        prev_hash = h

    return dec.reshape(image.shape)

# def decrypt_image(encrypted, key=b'secret_key', iv=b'init_vector'):
#     return encrypt_image(encrypted, key, iv)  # symmetric

# =============================
# Tampering (FIXED)
# =============================

def blackout_region_np(image, x_start, y_start, size):
    tampered = image.copy()

    h, w, _ = tampered.shape
    x_end = min(x_start + size, w)
    y_end = min(y_start + size, h)

    tampered[y_start:y_end, x_start:x_end] = 0
    return tampered

# =============================
# Metrics
# =============================

def entropy(image):
    hist = np.histogram(image.flatten(), bins=256, range=(0,256))[0]
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def npcr(a, b):
    return np.sum(a != b) / a.size * 100

def uaci(a, b):
    return np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255 * 100

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# =============================
# Plots
# =============================

def show_all(img, enc, enc_t, dec, dec_t):
    titles = ["Input", "Encrypted", "Tampered Enc", "Decrypted", "Tampered Dec"]
    images = [img, enc, enc_t, dec, dec_t]

    plt.figure(figsize=(15,6))
    for i in range(5):
        plt.subplot(2,3,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# =============================
# Main
# =============================

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Encrypt
    enc = encrypt_image(img)

    # Tamper encrypted
    enc_tamp = blackout_region_np(enc, 100, 100, 150)

    # Decrypt
    dec = decrypt_image(enc)
    dec_tamp = decrypt_image(enc_tamp)

    # =============================
    # Metrics comparison
    # =============================

    print("\n=== METRICS ===")

    print("Entropy:")
    print("Input:", entropy(img))
    print("Encrypted:", entropy(enc))

    print("\nNPCR (Input vs Encrypted):", npcr(img, enc))
    print("UACI (Input vs Encrypted):", uaci(img, enc))

    print("\nPSNR:")
    print("Input vs Decrypted:", psnr(img, dec))
    print("Input vs Tampered Decrypted:", psnr(img, dec_tamp))

    # =============================
    # Visualization
    # =============================

    show_all(img, enc, enc_tamp, dec, dec_tamp)

# =============================
# Run
# =============================

if __name__ == "__main__":
    process_image("Rome2.jpg")

# "If you want to shine like a sun, first burn like a sun" - Dr. APJ Abdul Kalam.
# Success is a continuous process
