# FastAPI Integration Guide for Hat Detection Frontend

## 1. Required FastAPI Setup

Add these imports to your FastAPI app:
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np
from PIL import Image
import io
```

## 2. Enable CORS

```python
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 3. Create the Prediction Endpoint

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives image, runs YOLO detection, returns results
    """
    # Read uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Convert to numpy array for YOLO
    img_array = np.array(image)
    
    # Run YOLO inference (adjust model path)
    from ultralytics import YOLO
    model = YOLO("path/to/your/model.pt")
    results = model(img_array)
    
    # Extract detections
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "box": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                "confidence": float(box.conf),
                "class": "hat"
            })
    
    # Draw bounding boxes on image
    annotated_image = results[0].plot()
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', annotated_image)
    image_base64 = base64.b64encode(buffer).decode()
    
    return {
        "detections": detections,
        "processed_image_base64": image_base64,
        "status": "success"
    }
```

## 4. Error Handling

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # validation
        if file.size > 10 * 1024 * 1024:  # 10MB
            return {"status": "error", "message": "File too large"}
        
        # ... rest of code
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
```

## 5. Run Locally for Testing

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend will hit: `http://localhost:8000/predict`

## 6. Environment Variables (for production)

Create `.env`:
```
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=./model/yolo26n.pt
```

## 7. Deployment Checklist

- [ ] Set specific CORS origins (not "*")
- [ ] Add authentication if needed
- [ ] Optimize model loading (cache after first load)
- [ ] Add request size limits
- [ ] Use production ASGI server (Uvicorn with multiple workers)
- [ ] Set up logging
- [ ] Add API documentation at `/docs`
