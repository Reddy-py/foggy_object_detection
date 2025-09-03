import os
import torchvision.transforms as T
import cv2
import time
import threading

import numpy as np
import torch
from flask import Flask, render_template, request, Response
from torch import nn
from ultralytics import YOLO

from dehaze_aodnet import AODNet

app = Flask(__name__)

# Global variable for the uploaded video path (for a single-user demo)
VIDEO_PATH = None

# Global frame buffers (will store JPEG-encoded frames)
current_original_frame = None
current_defog_frame = None
current_detection_frame = None

# Lock for thread-safe access to the global frames
thread_lock = threading.Lock()

# Thread for background video processing
video_thread = None

# Load the YOLO model globally
model = YOLO("yolov8n.pt")  # or your custom weights, e.g. "best.pt"

def remove_fog(frame):
    """
    Remove fog using a simple CLAHE-based technique.
    Converts the frame to LAB color space, applies CLAHE on the L-channel,
    and converts back to BGR.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    defogged = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return defogged
class DehazerAOD:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        # Create your model
        self.model = AODNet().to(device)
        # Load your trained weights (adjust map_location as needed)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.eval()

        # Basic transform: convert BGR->RGB, to Tensor, etc.
        self.transform = T.ToTensor()
        # If your model needs normalization, insert it here

    def dehaze(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Dehaze a single OpenCV BGR frame; return a BGR frame.
        """
        # Convert BGR -> RGB for PyTorch
        rgb_frame = bgr_frame[:, :, ::-1]

        # To Tensor (C,H,W) and add batch dimension
        inp = self.transform(rgb_frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(inp)  # shape: (B,3,H,W)

        # Convert back to numpy
        out = out.squeeze(0).cpu().numpy().transpose(1,2,0)
        # Optionally clamp/normalize to 0..1 or 0..255
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)

        # Convert RGB -> BGR
        bgr_out = out[:, :, ::-1]
        return bgr_out

def domain_adaptation_train_loop(detection_model, domain_classifier,
                                 clear_loader, foggy_loader,
                                 optimizer, domain_optimizer, alpha=1.0):
    """
    A single epoch training loop (conceptual).
    - detection_model: YOLO or other detection model
    - domain_classifier: domain classification head (w/ GRL)
    - clear_loader, foggy_loader: dataloaders
    - alpha: GRL scaling factor
    """
    detection_model.train()
    domain_classifier.train()

    # Example: assume equal size of clear_loader and foggy_loader
    for (clear_imgs, clear_labels), (foggy_imgs, _) in zip(clear_loader, foggy_loader):
        # 1) Forward pass with detection model on clear images
        #    (this is normal supervised detection training)
        optimizer.zero_grad()
        detection_loss, clear_feats = detection_model(clear_imgs, labels=clear_labels, return_features=True)

        # 2) Forward pass domain classifier on those same features
        domain_loss_clear = nn.CrossEntropyLoss()(domain_classifier(clear_feats, alpha),
                                                  torch.zeros(clear_feats.size(0), dtype=torch.long))

        total_loss_clear = detection_loss + domain_loss_clear
        total_loss_clear.backward()
        optimizer.step()

        # 3) For foggy images, we typically don’t have ground-truth labels,
        #    but we can do pseudo-labeling or purely domain classification.
        domain_optimizer.zero_grad()
        # Forward pass for domain classification only
        _, foggy_feats = detection_model(foggy_imgs, return_features=True)
        domain_loss_foggy = nn.CrossEntropyLoss()(domain_classifier(foggy_feats, alpha),
                                                  torch.ones(foggy_feats.size(0), dtype=torch.long))

        # If using pseudo-labeling:
        #  - Use detection_model to predict bounding boxes on foggy images
        #  - Filter high-confidence boxes -> pseudo-labels
        #  - Recompute detection_loss with these pseudo-labels
        # [Pseudo-label code not shown here]

        domain_loss_foggy.backward()
        domain_optimizer.step()

    return detection_model, domain_classifier
def process_video():
    """
    Background thread function that reads the video, processes each frame
    to generate three outputs, and updates global frame buffers.
    """
    global current_original_frame, current_defog_frame, current_detection_frame, VIDEO_PATH

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            # Optionally, you can restart the video if needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Encode original frame as JPEG
        ret1, buffer_original = cv2.imencode(".jpg", frame)
        if not ret1:
            continue
        original_bytes = buffer_original.tobytes()

        # Process defogged frame
        defog_frame = remove_fog(frame)
        ret2, buffer_defog = cv2.imencode(".jpg", defog_frame)
        if not ret2:
            continue
        defog_bytes = buffer_defog.tobytes()

        # Process detection frame using YOLO on the defogged frame
        results = model(defog_frame)
        detection_frame = results[0].plot()
        # Ensure the annotated frame is the same size as the defogged frame
        if (detection_frame.shape[0] != defog_frame.shape[0] or
                detection_frame.shape[1] != defog_frame.shape[1]):
            detection_frame = cv2.resize(detection_frame,
                                         (defog_frame.shape[1], defog_frame.shape[0]))
        ret3, buffer_detection = cv2.imencode(".jpg", detection_frame)
        if not ret3:
            continue
        detection_bytes = buffer_detection.tobytes()

        # Update the global frame buffers using a lock for thread safety
        with thread_lock:
            current_original_frame = original_bytes
            current_defog_frame = defog_bytes
            current_detection_frame = detection_bytes

        # Sleep briefly to control the processing frame rate (~30 FPS)
        time.sleep(0.03)
    cap.release()

def gen_frames_original():
    """
    MJPEG generator for the original video frames.
    """
    global current_original_frame
    while True:
        with thread_lock:
            frame = current_original_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

def gen_frames_defog():
    """
    MJPEG generator for the defogged video frames.
    """
    global current_defog_frame
    while True:
        with thread_lock:
            frame = current_defog_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

def gen_frames_detection():
    """
    MJPEG generator for the detection output frames.
    """
    global current_detection_frame
    while True:
        with thread_lock:
            frame = current_detection_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route("/", methods=["GET", "POST"])
def index():
    global VIDEO_PATH, video_thread
    if request.method == "POST":
        # Handle the uploaded video file.
        file = request.files.get("video")
        if not file or file.filename == '':
            return "No video selected. Please select a valid file.", 400

        # Save the uploaded video
        os.makedirs("uploads", exist_ok=True)
        upload_path = os.path.join("uploads", file.filename)
        file.save(upload_path)
        VIDEO_PATH = upload_path
        print(f"[INFO] Video uploaded: {VIDEO_PATH}")

        # Start the background video processing thread if not already running
        if video_thread is None or not video_thread.is_alive():
            video_thread = threading.Thread(target=process_video, daemon=True)
            video_thread.start()

        # Render the page indicating that a video has been uploaded.
        return render_template("index.html", video_uploaded=True)

    # GET request: Render the page.
    return render_template("index.html", video_uploaded=(VIDEO_PATH is not None))

@app.route("/video_feed_original")
def video_feed_original():
    """
    MJPEG streaming endpoint for the original video.
    """
    return Response(gen_frames_original(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_defog")
def video_feed_defog():
    """
    MJPEG streaming endpoint for the defogged video.
    """
    return Response(gen_frames_defog(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_detection")
def video_feed_detection():
    """
    MJPEG streaming endpoint for the detection output.
    """
    return Response(gen_frames_detection(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
