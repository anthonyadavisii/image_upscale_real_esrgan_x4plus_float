Performs high‑quality 4× image upscaling using the Real‑ESRGAN x4+ model using ONNX Runtime with Snapdragon NPU

Original (4.1 MB - 4032x3024):
![nature](https://github.com/user-attachments/assets/9b17ce47-65dc-459f-9c56-ae328b0decd7)
Upscaled (9.1 MB - 16128x12096:
![nature_x4_tiled](https://github.com/user-attachments/assets/9b222b4e-223d-4029-bd30-632568d0bf84)
Original (4.08 MB - 4032x3024):
![IMG_20191109_121400948](https://github.com/user-attachments/assets/f858fb53-d9b6-4f37-bdb6-e708acf3f26a)
Upscaled (7.37 MB - 16128x12096):
![IMG_20191109_121400948_x4_tiled](https://github.com/user-attachments/assets/2479f7e6-cef4-46db-ac8f-713107e0a17e)

Exported from Qualcomm AI hub using command in venv (Python 3.13.11 *AMD 64*):
python -m qai_hub_models.models.real_esrgan_x4plus.export --device "Snapdragon X Elite CRD" --target-runtime onnx

Tested in Python 3.12.10 (ARM64) venv.
