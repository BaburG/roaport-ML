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
import base64 # Added for image encoding
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

# Cloudflare AI Gateway Configuration
CLOUDFLARE_AI_GATEWAY_URL = os.getenv("CLOUDFLARE_AI_GATEWAY_URL", "https://gateway.ai.cloudflare.com/v1/e16d722126ccef480a24b7cc683d3e35/roaport-gateway/google-ai-studio/v1/models/gemini-2.0-flash:generateContent")
GOOGLE_AI_STUDIO_TOKEN = os.getenv("GOOGLE_AI_STUDIO_TOKEN")

# Flag to control consumer shutdown
should_continue = True
connection = None
channel = None
model = None

def image_to_base64(image_path):
    """Converts an image file to a base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def analyze_with_cloudflare_ai(image_base64, object_type):
    """Analyzes an image using Cloudflare AI Gateway for damaged signs or sidewalks."""
    if not GOOGLE_AI_STUDIO_TOKEN:
        logger.error("GOOGLE_AI_STUDIO_TOKEN is not set. Cannot analyze with Cloudflare AI.")
        return "Error: Missing API Token"

    headers = {
        'content-type': 'application/json',
        'x-goog-api-key': GOOGLE_AI_STUDIO_TOKEN
    }
    
    prompt = f"Is there a damaged {object_type} in this image? Respond with YES or NO"
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg", # Assuming JPEG, adjust if necessary
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(CLOUDFLARE_AI_GATEWAY_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # Raises an exception for bad status codes
        
        # Extract the text response from the AI
        # This might need adjustment based on the actual API response structure
        # Assuming the response structure is similar to a Gemini response
        response_data = response.json()
        if response_data.get("candidates") and len(response_data["candidates"]) > 0:
            content_parts = response_data["candidates"][0].get("content", {}).get("parts", [])
            if content_parts and len(content_parts) > 0:
                return content_parts[0].get("text", "No text response found.")
        return "Could not parse AI response."
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Cloudflare AI Gateway: {e}")
        return f"Error: API call failed - {e}"
    except Exception as e:
        logger.error(f"Unexpected error during Cloudflare AI analysis: {e}")
        return f"Error: Unexpected - {e}"

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

def connect_to_rabbitmq():
    """Establish connection to RabbitMQ with retry logic"""
    global connection, channel
    
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to connect to RabbitMQ (attempt {attempt + 1}/{max_retries})")
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    credentials=credentials,
                    heartbeat=600,  # 10 minutes heartbeat
                    connection_attempts=3,
                    retry_delay=2,
                    blocked_connection_timeout=300,  # 5 minutes
                )
            )
            channel = connection.channel()
            
            # Declare the queue
            channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
            
            # Set prefetch count to 1 to ensure fair dispatch
            channel.basic_qos(prefetch_count=1)
            
            logger.info("Successfully connected to RabbitMQ")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)  # Exponential backoff, max 30s
            else:
                logger.error("Max connection attempts reached")
                return False

def callback(ch, method, properties, body):
    """Process incoming messages with connection recovery"""
    # Check if we should continue processing or shut down
    if not should_continue:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        return
        
    try:
        message = json.loads(body)
        image_id = message.get('id')
        image_url = message.get('image_url')
        message_type = message.get('type') # Get the new 'type' field
        
        logger.info(f"Received message [{image_id}]: Type='{message_type}', URL={image_url}")
        
        if not message_type:
            logger.error(f"Message [{image_id}] is missing 'type' field. Rejecting.")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        start_time = time.time()
        image_path = None # Initialize image_path

        if message_type == "pothole":
            image_path = download_image(image_url)
            if image_path:
                pothole_exists = detect_pothole(image_path)
                result_text = "EXISTS" if pothole_exists else "NOT EXISTS"
                processing_time = time.time() - start_time
                logger.info(f"Result [{image_id} Pothole]: {result_text} (YOLO processed in {processing_time:.2f}s)")
            else:
                logger.warning(f"Skipping pothole detection for [{image_id}] due to download error.")

        elif message_type in ["sign", "sidewalk"]:
            image_path = download_image(image_url)
            if image_path:
                image_b64 = image_to_base64(image_path)
                if image_b64:
                    ai_response = analyze_with_cloudflare_ai(image_b64, message_type)
                    processing_time = time.time() - start_time
                    logger.info(f"Result [{image_id} {message_type.capitalize()}]: AI Response: '{ai_response}' (Cloudflare AI processed in {processing_time:.2f}s)")
                else:
                    logger.warning(f"Skipping AI analysis for [{image_id}] due to base64 conversion error.")
            else:
                logger.warning(f"Skipping AI analysis for [{image_id}] due to download error.")

        elif message_type == "none":
            logger.info(f"Message [{image_id}] is of type 'other'. Rejecting as per rule.")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            if image_path: os.unlink(image_path) # Clean up if downloaded before type check logic was hit
            return # Ensure no further processing or ack
        
        else:
            logger.error(f"Message [{image_id}] has unknown type: '{message_type}'. Rejecting.")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            if image_path: os.unlink(image_path) # Clean up if downloaded before type check logic was hit
            return # Ensure no further processing or ack

        if image_path:
            os.unlink(image_path) # General cleanup for downloaded files
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except (pika.exceptions.ConnectionClosedByBroker, 
            pika.exceptions.AMQPConnectionError,
            pika.exceptions.StreamLostError) as e:
        logger.error(f"Connection error while processing message: {e}")
        # Don't ack the message, let it be redelivered after reconnection
        # The connection will be handled by the main loop
        raise  # Re-raise to trigger reconnection in main loop
        
    except Exception as e:
        logger.error(f"Unhandled error processing message: {e}")
        # Ensure message is nacked in case of unexpected error before ack/nack
        try:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as ne:
            logger.error(f"Failed to NACK message after unhandled error: {ne}")

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
    
    if connection and not connection.is_closed:
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
        if not connect_to_rabbitmq():
            logger.error("Failed to connect to RabbitMQ. Consumer will not start.")
            return
        
        # Set up consumer
        channel.basic_consume(
            queue=RABBITMQ_QUEUE,
            on_message_callback=callback
        )
        
        logger.info(f"Consumer started. Waiting for messages on queue '{RABBITMQ_QUEUE}'")
        logger.info("To exit: press CTRL+C, send SIGTERM, or create a file 'shutdown.signal'")
        
        # Start consuming messages with shutdown check and connection recovery
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
            except (pika.exceptions.ConnectionClosedByBroker,
                    pika.exceptions.AMQPConnectionError,
                    pika.exceptions.StreamLostError,
                    ConnectionResetError) as e:
                logger.error(f"Connection lost: {e}")
                logger.info("Attempting to reconnect...")
                
                # Clean up current connection
                try:
                    if connection and not connection.is_closed:
                        connection.close()
                except:
                    pass
                connection = None
                channel = None
                
                # Try to reconnect
                if connect_to_rabbitmq():
                    # Set up consumer again
                    channel.basic_consume(
                        queue=RABBITMQ_QUEUE,
                        on_message_callback=callback
                    )
                    logger.info("Successfully reconnected and resumed consuming")
                else:
                    logger.error("Failed to reconnect. Exiting.")
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
