from PIL import Image, ImageChops
import numpy as np
import os

# Allow large images (safe for local QC)
Image.MAX_IMAGE_PIXELS = None

def prompt_for_image(prompt_text):
    path = input(prompt_text).strip('"').strip()
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        exit(1)
    return path

def enhance_delta(delta_img, mode):
    # Mode 1: No enhancement
    if mode == "1":
        return delta_img
    # Mode 2: Multiply differences
    if mode == "2":
        factor = 4  # You can adjust this
        return delta_img.point(lambda p: min(255, p * factor))
    # Mode 3: Auto-normalize
    if mode == "3":
        delta_np = np.array(delta_img).astype(np.float32)
        min_val = delta_np.min()
        max_val = delta_np.max()
        if max_val > min_val:
            delta_np = (delta_np - min_val) / (max_val - min_val) * 255.0
        return Image.fromarray(delta_np.astype(np.uint8))
    # Mode 4: Multiply + normalize
    if mode == "4":
        # Multiply first
        factor = 4
        amplified = delta_img.point(lambda p: min(255, p * factor))
        # Then normalize
        delta_np = np.array(amplified).astype(np.float32)
        min_val = delta_np.min()
        max_val = delta_np.max()
        if max_val > min_val:
            delta_np = (delta_np - min_val) / (max_val - min_val) * 255.0
        return Image.fromarray(delta_np.astype(np.uint8))
    print("Invalid mode selected. Using normal delta.")
    return delta_img

def compute_delta(original_path, upscaled_path, mode):
    # Load images
    orig = Image.open(original_path).convert("RGB")
    up = Image.open(upscaled_path).convert("RGB")
    # Resize original to match upscaled resolution
    orig_resized = orig.resize(up.size, Image.BICUBIC)
    # Compute pixel-wise difference image
    delta_img = ImageChops.difference(up, orig_resized)
    # Apply enhancement mode
    delta_img = enhance_delta(delta_img, mode)
    # Convert to numpy for numeric metrics
    orig_np = np.array(orig_resized).astype(np.float32)
    up_np = np.array(up).astype(np.float32)
    diff = np.abs(up_np - orig_np)
    print("\n=== QC Metrics ===")
    print(f"Original size: {orig.size}")
    print(f"Upscaled size: {up.size}")
    print(f"Mean absolute pixel difference: {diff.mean():.4f}")
    print(f"Max pixel difference: {diff.max():.4f}")
    # Show delta image
    print("\nDisplaying delta image (bright = more change)...")
    delta_img.show()

if __name__ == "__main__":
    print("=== Image Upscale QC Tool ===\n")
    original_path = prompt_for_image("Enter path to ORIGINAL image: ")
    upscaled_path = prompt_for_image("Enter path to UPSCALED image: ")
    print("\nSelect delta visualization mode:")
    print("  1 = Normal delta")
    print("  2 = Amplified delta (factor ×4)")
    print("  3 = Auto-normalized delta (adaptive contrast)")
    print("  4 = Amplified + Auto-normalized (maximum visibility)")
    mode = input("Enter mode number (1–4): ").strip()
    compute_delta(original_path, upscaled_path, mode)