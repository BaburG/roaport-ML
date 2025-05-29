import pika
import json
import random
import os
import sys
import argparse
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

links = [
    "https://img.roaport.com/5618a02e-91c8-409a-abc7-327fc2ed23d2.jpg",
    "https://img.roaport.com/d9e4b66d-f588-441f-bf13-d42373baa3b1.jpg",
    "https://img.roaport.com/901770ac-8f18-448a-b5cf-c094332387c0.jpg",
    "https://img.roaport.com/aa68d372-ce80-4301-a86c-07dfbbf4bf32.jpg",
    "https://img.roaport.com/b7aa3787-0cd2-4653-afc7-e6fee50464b7.jpg",
    "https://img.roaport.com/0aa6d568-6249-4149-862d-5c3bad79cdea.jpg",
    "https://img.roaport.com/bda01d3d-ebb3-4914-9b1a-d4eac0b52c82.jpg",
    "https://img.roaport.com/e0548ab0-95f7-45f6-8e58-5595e011f1a4.jpg",
    "https://img.roaport.com/f672e435-dfc3-42c7-9ef9-006dea507058.jpg",
]

# Get RabbitMQ configuration from environment variables
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "image_processing")

def generate_id() -> str:
    """Generate a random 4-digit ID"""
    return f"{random.randint(1000, 9999)}"

def get_random_link() -> str:
    """Get a random link from the list"""
    return random.choice(links)

def setup_rabbitmq_connection():
    """Set up and return RabbitMQ connection and channel"""
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
    
    return connection, channel

def send_message(channel, link: str, image_id: str) -> None:
    """Send a link and ID to RabbitMQ"""
    message = {
        'type' : 'pothole',
        'id': image_id,
        'image_url': link
    }
    
    # Publish message
    channel.basic_publish(
        exchange='',
        routing_key=RABBITMQ_QUEUE,
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        )
    )
    print(f"Sent {image_id}: {link}")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Send messages to RabbitMQ')
    parser.add_argument('-n', '--num_messages', type=int, default=10,
                      help='Number of messages to send (default: 10)')
    args = parser.parse_args()
    
    num_messages = args.num_messages
    
    print(f"Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}")
    connection, channel = setup_rabbitmq_connection()
    
    try:
        for i in range(num_messages):
            image_id = generate_id()
            link = get_random_link()
            send_message(channel, link, image_id)
        
        print(f"Successfully sent {num_messages} messages to RabbitMQ queue '{RABBITMQ_QUEUE}'")
    
    finally:
        connection.close()
        print("Connection closed")

if __name__ == "__main__":
    main()