import gradio as gr
import requests

API_UPLOAD = "http://127.0.0.1:8000/upload_video"
LIVE_URL = "http://127.0.0.1:8000/detect_video"


# ---------------- UPLOAD ----------------
def detect_video(video, target):
    if video is None:
        return None

    with open(video, "rb") as f:
        response = requests.post(
            API_UPLOAD,
            files={"file": ("video.mp4", f, "video/mp4")},
            data={"target": target}
        )

    output_path = "result.mp4"

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


# ---------------- LIVE ----------------
def start_live(target):
    if target:
        return f'<img src="{LIVE_URL}?target={target}" width="100%">'
    return f'<img src="{LIVE_URL}" width="100%">'


def stop_live():
    return ""


# ---------------- UI ----------------
with gr.Blocks() as demo:

    gr.Markdown("#  Object Detection Dashboard")

    mode = gr.Radio(
        ["Upload Video", "Live Camera"],
        value="Upload Video",
        label="Select Mode"
    )

    # -------- UPLOAD UI --------
    with gr.Column(visible=True) as upload_ui:
        with gr.Row():

            with gr.Column():
                gr.Markdown("###  Upload Video")
                video_input = gr.Video(height=350)

                target_input = gr.Textbox(label="Object")

                run_btn = gr.Button("Run Detection")

            with gr.Column():
                gr.Markdown("###  Output")
                output_video = gr.Video(height=350)

    # -------- LIVE UI --------
    with gr.Column(visible=False) as live_ui:
        with gr.Row():

            with gr.Column():
                gr.Markdown("###  Live Input")

                live_target = gr.Textbox(label="Object (optional)")

                start_btn = gr.Button("▶ Start")
                stop_btn = gr.Button("⏹ Stop")

            with gr.Column():
                gr.Markdown("###  Live Output")
                live_stream = gr.HTML()

    # -------- MODE SWITCH --------
    def switch_mode(choice):
        if choice == "Upload Video":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    mode.change(
        switch_mode,
        inputs=mode,
        outputs=[upload_ui, live_ui]
    )

    # -------- ACTIONS --------
    run_btn.click(
        detect_video,
        inputs=[video_input, target_input],
        outputs=output_video
    )

    start_btn.click(
        start_live,
        inputs=live_target,
        outputs=live_stream
    )

    stop_btn.click(
        stop_live,
        outputs=live_stream
    )


# 🔥 CORRECT CSS (SMALL ICONS, NOT REMOVED)
demo.launch(
    theme=gr.themes.Base(),
    css="""
    /* KEEP FOOTER BUT MAKE IT SMALL */
    footer {
        transform: scale(0.6);
        transform-origin: bottom center;
        opacity: 0.7;
    }

    /* reduce icon size */
    svg {
        max-height: 40px !important;
    }

    /* CLEAN WIDTH */
    .gradio-container {
        max-width: 100% !important;
        padding: 10px !important;
    }

    /* IMPORTANT: allow scroll */
    body {
        overflow: auto !important;
    }

    video, img {
        height: 350px !important;
        width: 100% !important;
        border-radius: 10px;
    }
    """
)