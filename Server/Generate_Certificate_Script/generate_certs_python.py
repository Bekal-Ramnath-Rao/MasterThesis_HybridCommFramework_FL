#!/usr/bin/env python3
"""
Generate self-signed SSL/TLS certificates using cryptography library
"""
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from datetime import datetime, timedelta
from pathlib import Path
import ipaddress

def generate_self_signed_cert():
    """Generate a self-signed certificate for localhost"""
    
    # Create certs directory
    certs_dir = Path("certs")
    certs_dir.mkdir(exist_ok=True)
    
    print("Generating self-signed certificate for QUIC...")
    
    # Generate private key
    print("1. Generating RSA private key (2048 bits)...")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Create certificate
    print("2. Creating self-signed certificate...")
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FL Organization"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("*.local"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256())
    
    # Write private key
    print("3. Saving private key...")
    key_file = certs_dir / "server-key.pem"
    with open(key_file, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # Write certificate
    print("4. Saving certificate...")
    cert_file = certs_dir / "server-cert.pem"
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    print(f"\n✓ Certificates generated successfully!")
    print(f"  Location: {certs_dir.absolute()}")
    print(f"  Certificate: {cert_file.name}")
    print(f"  Private Key: {key_file.name}")
    print(f"  Valid for: 365 days")

if __name__ == "__main__":
    try:
        generate_self_signed_cert()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import sys
        sys.exit(1)
