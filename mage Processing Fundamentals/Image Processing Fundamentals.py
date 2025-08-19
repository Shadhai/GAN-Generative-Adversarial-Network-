import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r'C:\Users\Shadh\Pictures\Camera Roll\gan.jpg')

# Check if image loaded correctly
if img is None:
    raise FileNotFoundError("Image not found or path is incorrect.")

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display images
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title('Original (BGR)')
plt.imshow(img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Grayscale')
plt.imshow(img_gray, cmap='gray')
plt.axis('off')

plt.savefig('output.png')  # Save plot as an image file
print("Plot saved as output.png")
