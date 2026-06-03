from ultralytics import YOLO
import csv

model = YOLO("yolov8n.pt")
FPS = 29.97
VIDEO_PATH = r"C:\Users\marti\match_test.mp4"  # cambia con il tuo percorso
MAX_SECONDS = None  # None = tutto il video, oppure es. 300 per 5 minuti

results = model.track(
    source=VIDEO_PATH,
    tracker="botsort.yaml",  # botsort è migliore di bytetrack per il Reid
    device="0",  # GPU, usa "cpu" se non hai GPU
    save=False,
    stream=True,
    classes=[0, 32],  # 0=persone, 32=pallone
    conf=0.4,
    verbose=False
)

with open("match_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "timestamp_sec", "type", "player_id", "x_center", "y_center", "width", "height", "confidence"])
    
    for frame_idx, result in enumerate(results):
        timestamp = round(frame_idx / FPS, 3)
        
        if MAX_SECONDS and timestamp > MAX_SECONDS:
            break
            
        if result.boxes is None:
            continue
            
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
            print(f"Frame {frame_idx} — {round(timestamp,0)}s — {round(frame_idx/35965*100, 1)}%")

print("FATTO! File: match_output.csv")