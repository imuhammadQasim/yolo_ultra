from pathlib import Path
from ultralytics import YOLO

# Resolve the absolute path to your project root (yolo_ultra)
BASE_DIR = Path(__file__).resolve().parent.parent

# Point to the correct folder path you just found
model_path = BASE_DIR / "model" / "runs" / "detect" / "license_plate_model" / "weights" / "best.pt"
image_path = BASE_DIR / "data" / "image.png"

# Load the model
model = YOLO(str(model_path))

# Run inference
result = model(str(image_path))

# Save results
result[0].save("number_plate_detection.png")
print("Inference successful! Image saved as number_plate_detection.png")
