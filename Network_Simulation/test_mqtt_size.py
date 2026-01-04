"""Test script to verify MQTT broker can handle large messages"""
import paho.mqtt.client as mqtt
import time
import sys

# Test message sizes
TEST_SIZES = [
    (10 * 1024 * 1024, "10 MB"),   # Should work with old config
    (15 * 1024 * 1024, "15 MB"),   # Should fail with old config (10MB), work with new (20MB)
    (20 * 1024 * 1024, "20 MB"),   # Should work only with new config
]

class MQTTTester:
    def __init__(self):
        self.received = False
        self.received_size = 0
        self.disconnect_code = None
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"✓ Connected to broker")
            client.subscribe("test/large_message", qos=0)
        else:
            print(f"✗ Connection failed with code {rc}")
            
    def on_message(self, client, userdata, msg):
        self.received = True
        self.received_size = len(msg.payload)
        print(f"✓ Received message: {self.received_size:,} bytes")
        
    def on_disconnect(self, client, userdata, rc):
        self.disconnect_code = rc
        if rc != 0:
            print(f"✗ Unexpected disconnect, return code {rc}")
            
    def on_publish(self, client, userdata, mid):
        print(f"✓ Publish confirmed (mid={mid})")

def test_message_size(size_bytes, size_label):
    print(f"\n{'='*60}")
    print(f"Testing {size_label} ({size_bytes:,} bytes)")
    print(f"{'='*60}")
    
    tester = MQTTTester()
    
    # Create client
    client = mqtt.Client(client_id="mqtt_size_tester", protocol=mqtt.MQTTv311)
    client._max_packet_size = 25 * 1024 * 1024  # 25MB client limit
    client.on_connect = tester.on_connect
    client.on_message = tester.on_message
    client.on_disconnect = tester.on_disconnect
    client.on_publish = tester.on_publish
    
    try:
        # Connect
        print("Connecting to broker...", end=" ")
        client.connect("localhost", 1883, 60)
        client.loop_start()
        time.sleep(1)
        
        # Create large message
        print(f"Creating {size_label} test message...", end=" ")
        large_message = b'X' * size_bytes
        print("Done")
        
        # Publish
        print(f"Publishing {size_label} message...", end=" ")
        result = client.publish("test/large_message", large_message, qos=0)
        print(f"Sent (result code={result.rc})")
        
        # Wait for message
        print("Waiting for message receipt...")
        time.sleep(3)
        
        # Check results
        if tester.received and tester.received_size == size_bytes:
            print(f"✓ SUCCESS: {size_label} message sent and received!")
            return True
        elif tester.disconnect_code == 7:
            print(f"✗ FAILED: Message too large (disconnect code 7)")
            print(f"  → Broker rejected the message (exceeds message_size_limit)")
            return False
        else:
            print(f"✗ FAILED: Message not received")
            if tester.disconnect_code:
                print(f"  → Disconnect code: {tester.disconnect_code}")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    finally:
        client.loop_stop()
        client.disconnect()
        time.sleep(0.5)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MQTT Large Message Size Test")
    print("="*60)
    print("\nThis test verifies the broker's message_size_limit setting")
    print("Expected config: message_size_limit 20971520 (20MB)\n")
    
    results = []
    for size_bytes, size_label in TEST_SIZES:
        success = test_message_size(size_bytes, size_label)
        results.append((size_label, success))
        time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for label, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {label}")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if results[0][1] and results[1][1] and results[2][1]:
        print("✓ Broker accepts 20MB+ messages")
        print("  → message_size_limit is set to 20MB AND service was restarted")
    elif results[0][1] and not results[1][1]:
        print("✗ Broker still limited to ~10MB")
        print("  → Mosquitto service needs to be RESTARTED as Administrator!")
        print("\n  Run in Admin PowerShell:")
        print("    net stop mosquitto")
        print("    net start mosquitto")
    else:
        print("? Unexpected results - check Mosquitto configuration")
    
    print("="*60 + "\n")
