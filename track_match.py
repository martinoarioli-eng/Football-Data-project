from ultralytics import YOLO
import csv

model = YOLO("yolov8n.pt")
FPS = 29.97

results = model.track(
    source=r"C:\Users\marti\match_test.mp4",
    tracker="bytetrack.yaml",
    save=False,
    stream=True,
    classes=[0, 32]
)

with open("match_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "timestamp_sec", "type", "player_id", "x_center", "y_center", "width", "height", "confidence"])
    
    for frame_idx, result in enumerate(results):
        if result.boxes is None:
            continue
        timestamp = round(frame_idx / FPS, 3)
        for box in result.boxes:
            if box.id is None:
                continue
            x, y, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])
            pid = int(box.id[0])
            cls = int(box.cls[0])
            obj_type = "ball" if cls == 32 else "player"
            writer.writerow([frame_idx, timestamp, obj_type, pid, round(x,2), round(y,2), round(w,2), round(h,2), round(conf,3)])
        
        if frame_idx % 500 == 0:
            print(f"Frame {frame_idx}/35965 — {round(frame_idx/35965*100, 1)}%")

print("FATTO! File: match_output.csv")