from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...), target: str = Form(...)):
    contents = await file.read()

    # Convert to image
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return Response(content=b"", media_type="image/jpeg")

    results = model(img)[0]

    target = target.lower().strip()

    found = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()

        if target in label:
            found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Encode safely
    success, buffer = cv2.imencode(".jpg", img)

    if not success:
        return Response(content=b"", media_type="image/jpeg")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")