from fastapi import FastAPI, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
import cv2
from ultralytics import YOLO

app = FastAPI()

# Load model
model = YOLO("yolov8n.pt")

# ------------------- 🎥 LIVE STREAM -------------------

def generate_frames(target: str = None):
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        frame = cv2.resize(frame, (640, 480))

        if frame_count % 3 != 0:
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        results = model(frame, imgsz=320)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if target and target.lower() not in label.lower():
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.get("/")
def home():
    return {"message": "Video Detection API Running"}


@app.get("/detect_video")
def video_feed(target: str = Query(None)):
    return StreamingResponse(
        generate_frames(target),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ------------------- 📤 VIDEO UPLOAD -------------------

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...), target: str = Form(...)):
    input_path = "input.mp4"
    output_path = "output.mp4"

    # Save file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)

    # 🔥 FIX 1: Handle FPS properly
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0 or fps != fps:   # handles NaN also
        fps = 20

    fps = float(fps)

    width, height = 640, 480

    # 🔥 FIX 2: Proper VideoWriter
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (640,480)
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame = cv2.resize(frame, (width, height))

        # Skip frames
        if frame_count % 5 != 0:
            out.write(frame)
            continue

        results = model(frame, imgsz=320)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()

            if target.lower() in label:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        out.write(frame)

    # 🔥 FIX 3: Proper release
    cap.release()
    out.release()

    # 🔥 FIX 4: Return only after closing writer
    return FileResponse(output_path, media_type="video/mp4")