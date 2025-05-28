import cv2
from ultralytics import YOLO


def detect_objects(image_path, model_name, conf_threshold=0.25):
    """
    Detects objects in an image using a YOLO model and displays the image
    with bounding boxes.
    
    Args:
        image_path: Path to the image
        model_name: Path to the YOLO model weights
        conf_threshold: Confidence threshold (0-1) for showing detections
    """
    model = YOLO(model_name)
    img = cv2.imread(image_path)
    # Resize the image to a standard size (e.g., 640x640)
    #img = cv2.resize(img, (640, 640))
    
    # Pass confidence threshold to the model
    results = model(img, conf=conf_threshold)
    
    annotated_img = results[0].plot()
    cv2.imshow("YOLO Object Detection", annotated_img)
    
    # Wait for a key press and check if it's 'q' or ESC (27)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q') or key == 27:
        cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define the path to the best weight and the image
    BEST_WEIGHT_PATH = "PotholeDetectionXL.pt"  # Replace with your actual best weight path
    IMAGE_PATH = "test1.jpg"  # Replace with your actual image path
    CONFIDENCE_THRESHOLD = 0.85  # Set your desired confidence threshold (0-1)

    detect_objects(IMAGE_PATH, BEST_WEIGHT_PATH, CONFIDENCE_THRESHOLD) 