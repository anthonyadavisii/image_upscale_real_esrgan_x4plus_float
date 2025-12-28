import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import math

# Your paths
model_path = r"C:\Users\v7dav\ai-env\models\Computer_Vision\real_esrgan_x4plus_float.onnx\job_j57xqwqng_optimized_onnx\model.onnx"
img_path   = r"C:\Users\v7dav\ai-env\Images\nature.jpg"

TILE = 128
OVERLAP = 16
SCALE = 4

def preprocess(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    arr = np.expand_dims(arr, 0)        # NCHW
    return arr

def postprocess(tensor):
    img = tensor.squeeze(0)
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = (img * 255.0).astype(np.uint8)
    return Image.fromarray(img)

def upscale_tile(sess, tile):
    inp = preprocess(tile)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    out = sess.run([output_name], {input_name: inp})[0]
    return postprocess(out)

def upscale_image_tiled(model_path, input_path):
    # Load model
    sess = ort.InferenceSession(
        model_path,
        providers=["QNNExecutionProvider", "CPUExecutionProvider"]
    )

    # Load image
    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    # Calculate tile grid
    tiles_x = math.ceil((w - OVERLAP) / (TILE - OVERLAP))
    tiles_y = math.ceil((h - OVERLAP) / (TILE - OVERLAP))

    # Prepare output canvas
    out_w = w * SCALE
    out_h = h * SCALE
    output = Image.new("RGB", (out_w, out_h))

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # Compute tile coordinates
            x0 = tx * (TILE - OVERLAP)
            y0 = ty * (TILE - OVERLAP)
            x1 = min(x0 + TILE, w)
            y1 = min(y0 + TILE, h)

            # Extract tile
            tile = img.crop((x0, y0, x1, y1))

            # Pad tile to 128×128 if needed
            if tile.size != (TILE, TILE):
                padded = Image.new("RGB", (TILE, TILE))
                padded.paste(tile, (0, 0))
                tile = padded

            # Upscale tile
            up = upscale_tile(sess, tile)

            # Compute output placement
            ox = x0 * SCALE
            oy = y0 * SCALE

            # Paste into output
            output.paste(up, (ox, oy))

    # Save result
    base, ext = os.path.splitext(input_path)
    out_path = f"{base}_x4_tiled{ext}"
    output.save(out_path)
    print(f"Upscaled image saved to: {out_path}")

# Run it
upscale_image_tiled(model_path, img_path)