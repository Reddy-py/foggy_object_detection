import cv2
from ultralytics import YOLO

def main():
    # 1. Load the YOLOv8 model
    model = YOLO("yolov8n.pt")  # You can replace this with your custom model path, e.g. "best.pt"

    # 2. Create a VideoCapture object to read from a video file
    video_path = "videoplayback.mp4"  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Optional: get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # 3. (Optional) Create a VideoWriter to save the output
    #    If you want to save the annotated video, uncomment the lines below
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
    # out = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames or error reading the video

        # 4. Inference with YOLOv8
        results = model(frame)

        # 5. Plot detections on the frame
        annotated_frame = results[0].plot()

        # 6. Display the annotated frame
        cv2.imshow("YOLOv8 Video Detection", annotated_frame)

        # (Optional) write the annotated frame to the output video
        # out.write(annotated_frame)

        # Press 'ESC' to exit the loop early
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Clean up
    cap.release()
    # out.release()  # if you used VideoWriter
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
