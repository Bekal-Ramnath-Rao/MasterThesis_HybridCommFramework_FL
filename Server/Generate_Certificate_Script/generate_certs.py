#!/usr/bin/env python3
"""
Generate self-signed SSL/TLS certificates for QUIC and gRPC protocols
"""
import os
import subprocess
from pathlib import Path

def generate_certificates():
    """Generate self-signed certificates for secure communication"""
    
    # Create certs directory
    certs_dir = Path("certs")
    certs_dir.mkdir(exist_ok=True)
    
    print("Generating SSL/TLS certificates...")
    
    # Generate private key
    print("1. Generating private key...")
    subprocess.run([
        "openssl", "genrsa",
        "-out", str(certs_dir / "server-key.pem"),
        "2048"
    ], check=True)
    
    # Generate certificate signing request
    print("2. Generating certificate signing request...")
    subprocess.run([
        "openssl", "req",
        "-new",
        "-key", str(certs_dir / "server-key.pem"),
        "-out", str(certs_dir / "server-csr.pem"),
        "-subj", "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    ], check=True)
    
    # Generate self-signed certificate
    print("3. Generating self-signed certificate...")
    subprocess.run([
        "openssl", "x509",
        "-req",
        "-days", "365",
        "-in", str(certs_dir / "server-csr.pem"),
        "-signkey", str(certs_dir / "server-key.pem"),
        "-out", str(certs_dir / "server-cert.pem")
    ], check=True)
    
    # Create a combined PEM file for QUIC
    print("4. Creating combined certificate file...")
    with open(certs_dir / "combined.pem", "w") as combined:
        with open(certs_dir / "server-cert.pem", "r") as cert:
            combined.write(cert.read())
        with open(certs_dir / "server-key.pem", "r") as key:
            combined.write(key.read())
    
    # Set appropriate permissions
    os.chmod(certs_dir / "server-key.pem", 0o600)
    os.chmod(certs_dir / "server-cert.pem", 0o644)
    os.chmod(certs_dir / "combined.pem", 0o600)
    
    print("\n✓ Certificates generated successfully!")
    print(f"  - Private key: {certs_dir / 'server-key.pem'}")
    print(f"  - Certificate: {certs_dir / 'server-cert.pem'}")
    print(f"  - Combined: {certs_dir / 'combined.pem'}")
    print("\nNote: These are self-signed certificates for development/testing only.")
    print("For production, use certificates from a trusted CA.\n")

if __name__ == "__main__":
    try:
        generate_certificates()
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error generating certificates: {e}")
        print("Make sure OpenSSL is installed and available in your PATH.")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        exit(1)
