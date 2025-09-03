import cv2
from ultralytics import YOLO


def main():
    # Initialize the YOLOv8 model (e.g., yolov8n, yolov8s, etc.)
    model = YOLO("yolov8n.pt")  # or "path/to/your_custom_model.pt"

    # Create a VideoCapture object to access your webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam; change if needed

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference with YOLOv8
        results = model(frame)  # Perform inference on the frame

        # Plot the detection results directly on the frame
        # results[0] is the first (and only) batch result in this scenario
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # Press 'ESC' to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
