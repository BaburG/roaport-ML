import pika
import json
import time
import os
import requests
import tempfile
import cv2
import signal
import sys
import logging
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('pothole-detector')

# Load environment variables from .env file
load_dotenv()

# Get RabbitMQ configuration from environment variables
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "image_processing")

# YOLO model path
YOLO_MODEL_PATH = "PotholeDetectionXL.pt"
CONFIDENCE_THRESHOLD = 0.85

# Flag to control consumer shutdown
should_continue = True
connection = None
channel = None
model = None

def detect_pothole(image_path, conf_threshold=CONFIDENCE_THRESHOLD):
    """
    Detects if a pothole exists in an image using YOLO model.
    Returns True if a pothole is detected, False otherwise.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return False
    
    results = model(img, conf=conf_threshold)
    
    # Check if 'pothole' is in the detected classes
    for r in results:
        for c in r.boxes.cls:
            # Get class name (assuming pothole is a class in the model)
            class_name = model.names[int(c)]
            if class_name.lower() == "pothole":
                return True
    
    return False

def download_image(url, timeout=30):
    """Download image from URL to a temporary file"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_file.name
    except Exception as e:
        logger.error(f"Error downloading image {url}: {e}")
        return None

def callback(ch, method, properties, body):
    """Process incoming messages"""
    # Check if we should continue processing or shut down
    if not should_continue:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        return
        
    try:
        message = json.loads(body)
        image_id = message.get('id')
        image_url = message.get('image_url')
        
        logger.info(f"Processing image [{image_id}] from {image_url}")
        
        # Download the image
        start_time = time.time()
        image_path = download_image(image_url)
        
        if image_path:
            # Detect if pothole exists
            pothole_exists = detect_pothole(image_path)
            
            # Print result
            result = "EXISTS" if pothole_exists else "NOT EXISTS"
            processing_time = time.time() - start_time
            logger.info(f"Result [{image_id}]: Pothole {result} (processed in {processing_time:.2f}s)")
            
            # Clean up the temporary file
            os.unlink(image_path)
        
        # Acknowledge the message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        # Negative acknowledgment in case of error
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def graceful_shutdown(signum=None, frame=None):
    """Handle graceful shutdown of the consumer"""
    global should_continue, channel, connection
    
    logger.info("Initiating graceful shutdown...")
    should_continue = False
    
    if channel:
        try:
            channel.stop_consuming()
        except Exception as e:
            logger.error(f"Error stopping consumption: {e}")
    
    if connection and connection.is_open:
        try:
            connection.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    logger.info("Consumer shutdown complete")
    
    # Exit if this was called from a signal handler
    if signum is not None:
        sys.exit(0)

def register_signal_handlers():
    """Register signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

def load_model():
    """Load and initialize the YOLO model with details logging"""
    global model
    
    logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    start_time = time.time()
    
    try:
        model = YOLO(YOLO_MODEL_PATH)
        load_time = time.time() - start_time
        
        # Get model details
        model_type = model.type if hasattr(model, 'type') else 'Unknown'
        model_task = model.task if hasattr(model, 'task') else 'Unknown'
        num_classes = len(model.names) if hasattr(model, 'names') else 0
        class_names = list(model.names.values()) if hasattr(model, 'names') else []
        
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        logger.info(f"Model details: Type={model_type}, Task={model_task}, Classes={num_classes}")
        logger.info(f"Detected classes: {', '.join(class_names)}")
        logger.info(f"Detection confidence threshold: {CONFIDENCE_THRESHOLD}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def main():
    global connection, channel, should_continue
    
    logger.info("Starting Pothole Detection Consumer")
    
    # Register signal handlers
    register_signal_handlers()
    should_continue = True
    
    # Load the YOLO model
    if not load_model():
        logger.error("Failed to initialize model. Consumer will not start.")
        return
    
    try:
        # Connect to RabbitMQ
        logger.info(f"Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}")
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials
            )
        )
        channel = connection.channel()
        
        # Declare the queue
        channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
        
        # Set prefetch count to 1 to ensure fair dispatch
        channel.basic_qos(prefetch_count=1)
        
        # Set up consumer
        channel.basic_consume(
            queue=RABBITMQ_QUEUE,
            on_message_callback=callback
        )
        
        logger.info(f"Consumer started. Waiting for messages on queue '{RABBITMQ_QUEUE}'")
        logger.info("To exit: press CTRL+C, send SIGTERM, or create a file 'shutdown.signal'")
        
        # Start consuming messages with shutdown check
        while should_continue:
            # Check for shutdown signal file
            if os.path.exists("shutdown.signal"):
                logger.info("Shutdown signal file detected")
                graceful_shutdown()
                break
                
            try:
                # Process messages, but timeout regularly to check shutdown conditions
                connection.process_data_events(time_limit=1.0)
            except KeyboardInterrupt:
                graceful_shutdown()
                break
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                time.sleep(1)  # Short delay before retry
                
    except KeyboardInterrupt:
        graceful_shutdown()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        graceful_shutdown()

if __name__ == "__main__":
    main()
