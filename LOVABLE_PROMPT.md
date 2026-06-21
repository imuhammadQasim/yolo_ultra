# Hat Detection Frontend - Lovable AI Prompt

## Project Overview
Build a responsive web frontend for a hat detection system using YOLOv8. Users upload images and receive real-time detection results showing hats identified with confidence scores and bounding boxes.

## Key Features
1. **Image Upload**
   - Drag-and-drop or click-to-upload interface
   - Support JPG, PNG, WebP formats
   - Preview uploaded image before processing
   - Max file size: 10MB

2. **Detection Display**
   - Show uploaded image with bounding boxes around detected hats
   - Display confidence score for each detection
   - Show total number of hats detected
   - Highlight detection areas with contrasting colors

3. **Processing Feedback**
   - Loading spinner while processing
   - Success/error messages
   - Processing time display

4. **Results Management**
   - Download annotated image with detections
   - Download results as JSON (coordinates, confidence scores)
   - Clear button to start new detection
   - View detection history (last 5 detections)

## Technical Integration
- **API Endpoint:** `POST /predict` - Sends image, receives `{detections: [{box: [x1,y1,x2,y2], confidence: float, class: "hat"}], processed_image_base64: string}`
- **Base URL:** Configurable via environment variable `VITE_API_URL` (default: `http://localhost:8000`)
- **Image Format:** Send as FormData with key "file"
- **Response:** Base64-encoded image + JSON detection data

## Design Requirements
- Clean, modern UI with dark/light mode toggle
- Mobile-responsive (mobile-first approach)
- Accessibility-friendly (WCAG 2.1 AA)
- Fast load time
- Intuitive user flow with minimal clicks

## Tech Stack
- React 18 + Vite
- TailwindCSS for styling
- Axios for API calls
- No authentication required

## Environment Setup
- Create `.env.local` with `VITE_API_URL=http://localhost:8000`
- API should have CORS enabled
