#!/usr/bin/env python3
"""Test AMQP message passing between client and server"""
import pika
import json
import time
import threading

# Test configuration
AMQP_HOST = 'localhost'
AMQP_PORT = 5672
CLIENT_ID = 99

def send_amqp_message():
    """Send a test message via AMQP"""
    time.sleep(2)  # Wait for consumer to start
    print("\n[TEST] Sending test message via AMQP...")
    
    try:
        credentials = pika.PlainCredentials('guest', 'guest')
        parameters = pika.ConnectionParameters(
            host=AMQP_HOST,
            port=AMQP_PORT,
            credentials=credentials,
            connection_attempts=3,
            retry_delay=1
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        # Declare exchange
        channel.exchange_declare(
            exchange='fl_client_updates',
            exchange_type='direct',
            durable=True
        )
        
        # Send test update
        test_message = {
            'client_id': CLIENT_ID,
            'round': 1,
            'weights': 'test_weights_data'
        }
        
        payload = json.dumps(test_message)
        channel.basic_publish(
            exchange='fl_client_updates',
            routing_key=f'client_{CLIENT_ID}_update',
            body=payload,
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        print(f"[TEST] Sent message: {test_message}")
        connection.close()
        
    except Exception as e:
        print(f"[TEST] Error sending: {e}")

def consume_amqp_messages():
    """Consume test messages via AMQP"""
    print("[TEST] Starting AMQP consumer...")
    
    try:
        credentials = pika.PlainCredentials('guest', 'guest')
        parameters = pika.ConnectionParameters(
            host=AMQP_HOST,
            port=AMQP_PORT,
            credentials=credentials,
            connection_attempts=3,
            retry_delay=1
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        # Declare exchange
        channel.exchange_declare(
            exchange='fl_client_updates',
            exchange_type='direct',
            durable=True
        )
        
        # Create test queue
        test_queue = f'client_{CLIENT_ID}_updates'
        channel.queue_declare(queue=test_queue, durable=True)
        channel.queue_bind(
            exchange='fl_client_updates',
            queue=test_queue,
            routing_key=f'client_{CLIENT_ID}_update'
        )
        
        print(f"[TEST] Polling queue '{test_queue}' for 10 seconds...")
        
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < 10:
            method, properties, body = channel.basic_get(queue=test_queue, auto_ack=True)
            if body:
                message_count += 1
                try:
                    data = json.loads(body.decode())
                    print(f"[TEST] ✓ Received message #{message_count}: {data}")
                except Exception as e:
                    print(f"[TEST] Error parsing message: {e}")
            time.sleep(0.5)
        
        if message_count == 0:
            print("[TEST] ✗ NO MESSAGES RECEIVED - AMQP polling not working!")
        else:
            print(f"[TEST] ✓ Successfully received {message_count} message(s)")
        
        connection.close()
        
    except Exception as e:
        print(f"[TEST] Consumer error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("=" * 60)
    print("AMQP MESSAGE PASSING TEST")
    print("=" * 60)
    
    # Start consumer in background
    consumer_thread = threading.Thread(target=consume_amqp_messages, daemon=False)
    consumer_thread.start()
    
    # Start sender
    sender_thread = threading.Thread(target=send_amqp_message, daemon=False)
    sender_thread.start()
    
    # Wait for completion
    consumer_thread.join()
    sender_thread.join()
    
    print("\n[TEST] Test complete")
