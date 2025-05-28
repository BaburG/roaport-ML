# RabbitMQ Image Processing Queue

This project demonstrates how to use RabbitMQ to process image URLs.

## Setup

1. Install the required packages:
```
pip install -r requirements.txt
```

2. Create a `.env` file with your RabbitMQ connection details:
```
# RabbitMQ Connection Settings
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASS=guest
RABBITMQ_QUEUE=image_processing
```

## Usage

### Sending Messages

Send random image URLs to the queue:

```
python test_producer.py -n 20
```

This will send 20 random messages to the queue. Each message contains:
- A random 4-digit ID
- A randomly selected image URL from the list

Options:
- `-n`, `--num_messages`: Number of messages to send (default: 10)

### Receiving Messages

Start the consumer to process incoming messages:

```
python test_consumer.py
```

The consumer will:
- Connect to the RabbitMQ server
- Listen for messages on the configured queue
- Process each message (currently just prints the ID and URL)
- Acknowledge the message when processing is complete

Press Ctrl+C to stop the consumer. 