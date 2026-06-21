# from pathlib import Path
# from ultralytics import YOLO

# # Resolve the absolute path to your project root (yolo_ultra)
# BASE_DIR = Path(__file__).resolve().parent.parent

# # Point to the correct folder path you just found
# model_path = BASE_DIR / "model" / "runs" / "detect" / "license_plate_model" / "weights" / "best.pt"
# image_path = BASE_DIR / "data" / "image.png"

# # Load the model
# model = YOLO(str(model_path))

# # Run inference
# result = model(str(image_path))

# # Save results
# result[0].save("number_plate_detection.png")
# print("Inference successful! Image saved as number_plate_detection.png")

from pathlib import Path
from ultralytics import YOLO
import cv2
import easyocr

# Initialize EasyOCR reader (English language)
# This will download the OCR model components on its first run automatically
ocr_reader = easyocr.Reader(['en'], gpu=False) # Change gpu=False if you don't have a CUDA GPU

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "model" / "runs" / "detect" / "license_plate_model" / "weights" / "best.pt"
video_path = str(BASE_DIR / "data" / "traffic_video.mp4")  
output_path = "license_plate_output.mp4"

# Load YOLO model
model = YOLO(str(model_path))

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video specs
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Track unique/latest license plates to display on the sidebar
detected_plates_history = []

print("Processing video with Text Recognition... Click the window and press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  

    # Run detection
    results = model(frame, verbose=False)
    
    # Get the annotated frame from YOLO base system
    annotated_frame = results[0].plot()

    # Create a translucent sidebar overlay on the left side (width: 320px)
    sidebar_width = 320
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (sidebar_width, frame_height), (20, 20, 20), -1)
    # Blend the overlay with the original frame (0.6 opacity)
    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

    # Process bounding boxes to read text
    boxes = results[0].boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Crop the license plate out of the raw frame
        cropped_plate = frame[y1:y2, x1:x2]
        
        if cropped_plate.size > 0:
            # Convert to grayscale for better OCR accuracy
            gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
            
            # Read text from the cropped area
            ocr_result = ocr_reader.readtext(gray_plate)
            
            for (bbox, text, prob) in ocr_result:
                # Filter out low-confidence readings and clean spacing
                cleaned_text = text.upper().strip().replace(" ", "")
                if prob > 0.4 and len(cleaned_text) > 3:
                    if cleaned_text not in detected_plates_history:
                        detected_plates_history.append(cleaned_text)
                        # Keep only the 5 most recent records
                        if len(detected_plates_history) > 5:
                            detected_plates_history.pop(0)

    # Draw the text logs on the left sidebar
    cv2.putText(annotated_frame, "DETECTED PLATES:", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    
    y_offset = 90
    for idx, plate_num in enumerate(reversed(detected_plates_history)):
        display_str = f"{idx+1}. {plate_num}"
        # Draw background bubble for text visibility
        cv2.rectangle(annotated_frame, (15, y_offset - 25), (sidebar_width - 15, y_offset + 10), (40, 40, 40), -1)
        # Draw text
        cv2.putText(annotated_frame, display_str, (25, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 50

    # Save and show current frame
    out.write(annotated_frame)
    cv2.imshow("YOLO + OCR License Plate System", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished! Processed video saved as: {output_path}")
