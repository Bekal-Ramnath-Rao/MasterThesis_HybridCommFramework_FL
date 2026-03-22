#!/usr/bin/env python3
"""
TCP proxy so that AMQP clients in network namespaces can reach RabbitMQ on the host.

RabbitMQ often listens only on 127.0.0.1:5672. Processes in namespaces connect to the
bridge gateway (e.g. 10.200.0.254); the host receives that traffic. This proxy listens
on 0.0.0.0:listen_port on the host and forwards to 127.0.0.1:5672, so namespaces can
connect to gateway:listen_port and reach RabbitMQ without changing RabbitMQ config.

Usage: python3 amqp_proxy.py [listen_port] [bind_address]
  Default listen_port is 25673.
  bind_address: optional (e.g. 10.200.0.254). If set, listen only on this IP so the
  host accepts connections from the bridge; if omitted, listen on 0.0.0.0.
"""

import socket
import sys
import threading

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 5672
DEFAULT_LISTEN_PORT = 25673


def relay(client_sock: socket.socket, backend_sock: socket.socket) -> None:
    try:
        while True:
            data = client_sock.recv(65536)
            if not data:
                break
            backend_sock.sendall(data)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        try:
            client_sock.close()
        except OSError:
            pass
        try:
            backend_sock.close()
        except OSError:
            pass


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LISTEN_PORT
    bind_addr = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((bind_addr, port))
    except OSError as e:
        print(f"[amqp_proxy] Failed to bind {bind_addr}:{port}: {e}", file=sys.stderr)
        return 1
    server.listen(64)
    print(f"[amqp_proxy] Listening on {bind_addr}:{port} -> {BACKEND_HOST}:{BACKEND_PORT}", flush=True)

    while True:
        try:
            client_sock, _ = server.accept()
        except OSError:
            break
        try:
            backend_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            backend_sock.settimeout(300)
            backend_sock.connect((BACKEND_HOST, BACKEND_PORT))
        except OSError:
            client_sock.close()
            continue
        t1 = threading.Thread(target=relay, args=(client_sock, backend_sock), daemon=True)
        t2 = threading.Thread(target=relay, args=(backend_sock, client_sock), daemon=True)
        t1.start()
        t2.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
