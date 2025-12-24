"""
Generate self-signed certificates for QUIC server
Run this script in the Server directory to create cert.pem and key.pem
"""

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
from pathlib import Path

def generate_certificates():
    """Generate self-signed certificate and private key for QUIC server"""
    
    print("Generating private key...")
    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
        backend=default_backend()
    )
    
    print("Generating certificate...")
    # Generate certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Federated Learning"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, u"QUIC Server"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(u"localhost"),
            x509.DNSName(u"127.0.0.1"),
        ]),
        critical=False,
    ).sign(key, hashes.SHA256(), default_backend())
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Write private key
    key_path = script_dir / "key.pem"
    print(f"Writing private key to {key_path}...")
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # Write certificate
    cert_path = script_dir / "cert.pem"
    print(f"Writing certificate to {cert_path}...")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    print("\n" + "="*60)
    print("✓ Certificates generated successfully!")
    print("="*60)
    print(f"Private Key: {key_path}")
    print(f"Certificate: {cert_path}")
    print(f"Valid for: 365 days")
    print("\nYou can now run the QUIC server.")
    print("="*60)

if __name__ == "__main__":
    generate_certificates()
