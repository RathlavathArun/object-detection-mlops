import gradio as gr
import requests
import numpy as np
import cv2

API_URL = "http://127.0.0.1:8000/detect"

def detect(image, target):
    if image is None or target == "":
        return image

    # Encode image
    _, img_encoded = cv2.imencode(".jpg", image)

    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")},
            data={"target": target}
        )
    except:
        return image  # API not running

    if response.status_code != 200:
        return image

    # Decode response
    img_array = np.frombuffer(response.content, np.uint8)
    output_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return output_img


interface = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy"),
        gr.Textbox(label="Enter object (person, car, dog)")
    ],
    outputs=gr.Image(type="numpy"),
    title="YOLO + FastAPI Object Detection"
)

interface.launch()