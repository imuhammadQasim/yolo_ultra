from ultralytics import YOLO

model = YOLO("model/yolo26n.pt")

results = model.train(
    data="data/hat_dataset/data.yaml",
    epochs=5,
    project="./runs",
    name="hat_experiment",
    exist_ok=True,
    imgsz=640
)