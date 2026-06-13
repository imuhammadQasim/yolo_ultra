# from ultralytics import YOLO

# # Load a model
# model = YOLO('model/yolo26n.pt') 
# results = model("https://ultralytics.com/images/bus.jpg")

# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     for box in boxes:
#         confidence = box.conf.item()
#         class_id = box.cls.item()
#         class_name = model.names[class_id]
#         x1 , y1 , x2 , y2 = box.xyxy[0]
#         x,y,w,h = box.xywh[0]
        
#         print(
#             f"Class: {class_name}, "
#             f"Confidence: {confidence:.2f}, "
#             f"Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
#             f"Center Box: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})"
#         )
#     result.show()
#     result.save('data/result.jpg')


from ultralytics import YOLO
import cv2

model = YOLO("model/yolo26n.pt")

img = cv2.imread('person.jpg')
results = model(img) 
print(results)
for result in results:
    boxes = result.boxes
    for box in boxes:
        confidence = box.conf.item()
        class_id = box.cls.item()
        class_name = model.names[class_id]
        x1 , y1 , x2 , y2 = map(int, box.xyxy[0])
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label =  f"{class_name} {confidence:.2f}"
        cv2.putText(img, label,(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    
cv2.imshow('Img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("data/result.jpg", img)