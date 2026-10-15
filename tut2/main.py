import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Load image
img = cv2.imread('./img.png', cv2.IMREAD_GRAYSCALE)

# 2. Compute 2D Fourier Transform
f_transform = np.fft.fft2(img)
f_shifted = np.fft.fftshift(f_transform)

# Get image dimensions and find the center
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Filter radius (adjust this to change the effect intensity)
radius = 40

# ==========================================
# Task A: Create a Smoothed Image (Low-Pass)
# ==========================================
# Create a mask of zeros, with a filled circle of 1s at the center
mask_lp = np.zeros((rows, cols), np.uint8)
cv2.circle(mask_lp, (ccol, crow), radius, 1, thickness=-1)

# Apply mask and inverse DFT
f_shifted_lp = f_shifted * mask_lp
f_ishift_lp = np.fft.ifftshift(f_shifted_lp)
img_smoothed = np.abs(np.fft.ifft2(f_ishift_lp))

# ==========================================
# Task B: Detect Edges (High-Pass)
# ==========================================
# Create a mask of ones, with a filled circle of 0s at the center
mask_hp = np.ones((rows, cols), np.uint8)
cv2.circle(mask_hp, (ccol, crow), radius, 0, thickness=-1)

# Apply mask and inverse DFT
f_shifted_hp = f_shifted * mask_hp
f_ishift_hp = np.fft.ifftshift(f_shifted_hp)
img_edges = np.abs(np.fft.ifft2(f_ishift_hp))

# ==========================================
# 4. Display results
# ==========================================
plt.figure(figsize=(15, 5))

plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(img_smoothed, cmap='gray')
plt.title('Smoothed Image (Low Pass)'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(img_edges, cmap='gray')
plt.title('Edges (High Pass)'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("./frequency_domain_filters.png")
plt.show()