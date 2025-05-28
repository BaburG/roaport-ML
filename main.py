import cv2
from ultralytics import YOLO

def main():
    # 1) Load your YOLO model
    # Replace 'yolov8n.pt' with your weights file if using a custom name or path
    model = YOLO('yolo11m.pt')

    # 2) Initialize the webcam
    cap = cv2.VideoCapture(0)  # '0' is usually the default camera

    if not cap.isOpened():
        print("Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # 3) Run object detection
        results = model(frame, stream=True)  # YOLOv8’s streaming inference

        # 4) Parse detection results and draw bounding boxes
        for r in results:
            # Each 'r' is a result for the frame
            boxes = r.boxes  # Boxes object for detections
            # YOLOv8 returns bounding boxes along with confidence and class
            for box in boxes:
                # box.xyxy[0] is [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0]
                # box.cls is the class index, box.conf is confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Draw rectangle and label on the frame
                cv2.rectangle(frame, 
                              (int(x1), int(y1)), 
                              (int(x2), int(y2)), 
                              (0, 255, 0), 2)
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, 
                            (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2)

        # 5) Show the frame
        cv2.imshow("YOLO Real-Time Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6) Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()