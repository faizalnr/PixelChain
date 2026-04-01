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
# PixelChain Encryption
# =============================

def encrypt_image(image, key=b'secret_key', iv=b'init_vector'):
    flat = image.flatten()
    encrypted = np.zeros_like(flat)

    prev_hash = sha256(key + iv)

    for i in range(len(flat)):
        h_i = sha256(prev_hash + key + i.to_bytes(4, 'big'))
        ks = key_stream(h_i, 1)[0]
        encrypted[i] = flat[i] ^ ks
        prev_hash = h_i

    return encrypted.reshape(image.shape)


# =============================
# Decryption
# =============================

def decrypt_image(encrypted, key=b'secret_key', iv=b'init_vector'):
    flat = encrypted.flatten()
    decrypted = np.zeros_like(flat)

    prev_hash = sha256(key + iv)

    for i in range(len(flat)):
        h_i = sha256(prev_hash + key + i.to_bytes(4, 'big'))
        ks = key_stream(h_i, 1)[0]
        decrypted[i] = flat[i] ^ ks
        prev_hash = h_i

    return decrypted.reshape(encrypted.shape)


# =============================
# Analysis Functions
# =============================

def entropy(image):
    hist = np.histogram(image.flatten(), bins=256, range=(0,256))[0]
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def npcr(original, encrypted):
    return np.sum(original != encrypted) / original.size * 100


def uaci(original, encrypted):
    return np.mean(np.abs(original - encrypted)) / 255 * 100


def psnr(original, compared):
    mse = np.mean((original.astype(np.float64) - compared.astype(np.float64)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


def normalize(values):
    v = np.array(values, dtype=float)
    if np.max(v) == np.min(v):
        return v
    return (v - np.min(v)) / (np.max(v) - np.min(v))


# =============================
# Plot Functions
# =============================

def plot_metric(values, labels, title, save_path):
    norm_vals = normalize(values)

    plt.figure(figsize=(6,4))
    plt.bar(labels, norm_vals)

    for i, v in enumerate(values):
        plt.text(i, norm_vals[i] + 0.02, f"{v:.3f}", ha='center')

    plt.title(title)
    plt.ylabel("Normalized Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================
# Histogram Plot
# =============================

def plot_histograms(img, enc, dec, save_path):
    plt.figure(figsize=(18,5))

    plt.subplot(1,3,1)
    plt.hist(img.flatten(), bins=256)
    plt.title("Input Histogram")

    plt.subplot(1,3,2)
    plt.hist(enc.flatten(), bins=256)
    plt.title("Encrypted Histogram")

    plt.subplot(1,3,3)
    plt.hist(dec.flatten(), bins=256)
    plt.title("Decrypted Histogram")

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================
# Correlation Plot
# =============================

def correlation_scatter(image, num_samples=5000):
    img = image[:,:,0]
    x = img[:, :-1].flatten()
    y = img[:, 1:].flatten()

    idx = np.random.choice(len(x), size=min(num_samples, len(x)), replace=False)
    return x[idx], y[idx]


def plot_correlation(img, enc, dec, save_path):
    plt.figure(figsize=(18,5))

    x1, y1 = correlation_scatter(img)
    x2, y2 = correlation_scatter(enc)
    x3, y3 = correlation_scatter(dec)

    plt.subplot(1,3,1)
    plt.scatter(x1, y1, s=1)
    plt.title("Input Correlation")

    plt.subplot(1,3,2)
    plt.scatter(x2, y2, s=1)
    plt.title("Encrypted Correlation")

    plt.subplot(1,3,3)
    plt.scatter(x3, y3, s=1)
    plt.title("Decrypted Correlation")

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================
# Main Pipeline
# =============================

def process_image(image_path):
    name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(name, exist_ok=True)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    enc = encrypt_image(img)
    dec = decrypt_image(enc)

    # Save images
    cv2.imwrite(f"{name}/input.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{name}/encrypted.png", cv2.cvtColor(enc, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{name}/decrypted.png", cv2.cvtColor(dec, cv2.COLOR_RGB2BGR))

    # Metrics
    ent = [entropy(img), entropy(enc), entropy(dec)]
    npcr_val = [0, npcr(img, enc), 0]   # NPCR only meaningful for encrypted
    uaci_val = [0, uaci(img, enc), 0]
    psnr_val = [
        psnr(img, img),
        psnr(img, enc),
        psnr(img, dec)
    ]

    labels = ["Input", "Encrypted", "Decrypted"]

    # Plots
    plot_histograms(img, enc, dec, f"{name}/histograms.png")
    plot_correlation(img, enc, dec, f"{name}/correlation.png")

    plot_metric(ent, labels, "Entropy Comparison", f"{name}/entropy.png")
    plot_metric(npcr_val, labels, "NPCR Comparison", f"{name}/npcr.png")
    plot_metric(uaci_val, labels, "UACI Comparison", f"{name}/uaci.png")
    plot_metric(psnr_val, labels, "PSNR Comparison", f"{name}/psnr.png")

    # Excel
    df = pd.DataFrame({
        "Metric": ["Entropy", "NPCR", "UACI", "PSNR"],
        "Input": [ent[0], npcr_val[0], uaci_val[0], psnr_val[0]],
        "Encrypted": [ent[1], npcr_val[1], uaci_val[1], psnr_val[1]],
        "Decrypted": [ent[2], npcr_val[2], uaci_val[2], psnr_val[2]]
    })

    df.to_excel(f"{name}/analysis.xlsx", index=False)

    print("✅ All results saved in:", name)


# =============================
# Run
# =============================

if __name__ == "__main__":
    process_image("Rome1.jpg")

# "If you want to shine like a sun, first burn like a sun" - Dr. APJ Abdul Kalam.
# Success is a continuous process