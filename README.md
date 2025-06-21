# Roaport ML Service

This repository contains the Roaport Machine Learning (ML) service, a Python-based application designed to asynchronously process and verify road hazard images. It acts as a consumer for a RabbitMQ queue, automatically analyzing new reports submitted through the Roaport platform to reduce the need for manual verification.

## Features

- **Asynchronous Processing**: Consumes messages from a **RabbitMQ** queue, allowing it to process images independently of the main application flow and ensuring that user uploads are fast and responsive.
- **Hybrid ML Strategy**:
  - **Pothole Detection**: Utilizes a custom-trained **YOLOv8** model (`PotholeDetectionXL.pt`) for high-accuracy, specialized pothole detection in images.
  - **Sign & Sidewalk Classification**: Leverages the **Google Gemini API** via the **Cloudflare AI Gateway** for classifying damage to traffic signs and sidewalks. This pragmatic approach uses a powerful, general-purpose Vision-Language Model to handle categories where specific training data was unavailable.
- **Automated Verification**: If a hazard is detected with high confidence (either by YOLO or Gemini), the service automatically sends a verification request to the main backend (`www.roaport.com/api/verify`).
- **Robust Connection Handling**: Implements a resilient connection mechanism to RabbitMQ with automatic retries and exponential backoff to handle network interruptions gracefully.
- **Image Handling**: Downloads images from remote URLs (Cloudflare R2), processes them locally, and cleans up temporary files after analysis.
- **Container-Ready**: Designed to be run as a standalone service, easily containerized for deployment.

## Tech Stack

- **Language**: [Python](https://www.python.org/)
- **ML/Computer Vision**:
  - [Ultralytics YOLOv8](https://ultralytics.com/) for object detection.
  - [OpenCV](https://opencv.org/) for image processing.
  - [Google Gemini API](https://ai.google.dev/models/gemini) for general image classification.
- **Message Queue Client**: [Pika](https://pika.readthedocs.io/en/stable/) for RabbitMQ communication.
- **HTTP Client**: [Requests](https://requests.readthedocs.io/en/latest/) for downloading images and calling external APIs.
- **Environment Management**: [python-dotenv](https://pypi.org/project/python-dotenv/)

## Workflow

1.  The service establishes a persistent connection to the RabbitMQ server and listens for messages on the `image_processing` queue.
2.  When a new message arrives, the service consumes it. The message contains the `report_id`, `image_url`, and `type` of the hazard.
3.  The image is downloaded from the provided `image_url`.
4.  **If the `type` is "pothole"**:
    - The image is passed to the local YOLOv8 model.
    - If a pothole is detected with a confidence score above the threshold (85%), the report is considered verified.
5.  **If the `type` is "sign" or "sidewalk"**:
    - The image is encoded to Base64.
    - A carefully crafted prompt is sent along with the image to the Gemini API (e.g., *"Examine this image carefully for sidewalk damage... Respond with YES or NO."*).
    - If the API responds with "YES," the report is considered verified.
6.  If a report is successfully verified by either method, the service makes a `POST` request to the `/api/verify` endpoint of the main Roaport backend, passing the `report_id`.
7.  The main backend then updates the report's status in the database and sends a push notification to the user.
8.  Finally, the service sends an acknowledgment (`ack`) to RabbitMQ to confirm the message has been successfully processed and can be removed from the queue.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Access to a RabbitMQ server
- A Google AI Studio API Token (for Gemini)
- The YOLOv8 model weights file (`PotholeDetectionXL.pt`) placed in the root directory.

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# RabbitMQ Connection Settings
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASS=guest
RABBITMQ_QUEUE=image_processing

# Cloudflare AI Gateway / Google Gemini API
CLOUDFLARE_AI_GATEWAY_URL=<your_gateway_url>
GOOGLE_AI_STUDIO_TOKEN=<your_google_ai_studio_api_key>
```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/roaport-ML.git
    cd roaport-ML
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Service

To start the consumer and begin processing messages from the queue, run:

```bash
python prod_consumer.py
```

The service will log its status, connections, and processing results to the console. To stop the service gracefully, press `Ctrl+C`.
