"""
Security module for Google IAM, SSL, and authentication management
"""
import ssl
import os
import hashlib
import hmac
import jwt
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import secrets
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, session
from config import config

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Handle data encryption and decryption"""
    
    def __init__(self, encryption_key: str = None):
        self.key = encryption_key or config.security.encryption_key
        if not self.key:
            self.key = Fernet.generate_key().decode()
            logger.warning("No encryption key provided, generated new key")
        
        # Convert string key to bytes if needed
        if isinstance(self.key, str):
            self.key = self.key.encode()
        
        self.cipher = Fernet(base64.urlsafe_b64encode(self.key[:32]))
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            return ""
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            return ""
    
    def hash_password(self, password: str, salt: str = None) -> Dict[str, str]:
        """Hash password with salt"""
        if not salt:
            salt = secrets.token_hex(16)
        
        # Create hash
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return {
            'hash': base64.b64encode(password_hash).decode(),
            'salt': salt
        }
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        try:
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            
            return hmac.compare_digest(
                base64.b64encode(password_hash).decode(),
                stored_hash
            )
        except Exception as e:
            logger.error(f"Password verification failed: {str(e)}")
            return False

class JWTManager:
    """Handle JWT token generation and validation"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or config.security.jwt_secret
        self.algorithm = 'HS256'
        self.expiration_time = config.security.session_timeout  # seconds
    
    def generate_token(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JWT token"""
        try:
            payload = {
                'user_data': user_data,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(seconds=self.expiration_time)
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            return {
                'token': token,
                'expires_at': payload['exp'].isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            return {
                'valid': True,
                'user_data': payload.get('user_data'),
                'expires_at': payload.get('exp')
            }
            
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token has expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def refresh_token(self, token: str) -> Dict[str, Any]:
        """Refresh JWT token if valid"""
        verification = self.verify_token(token)
        
        if verification['valid']:
            return self.generate_token(verification['user_data'])
        else:
            return verification

class GoogleIAMManager:
    """Google Cloud IAM integration"""
    
    def __init__(self):
        self.credentials = None
        self.project_id = config.cloud.google_project_id
        self.credentials_path = config.cloud.google_credentials_path
        
        self.initialize_credentials()
    
    def initialize_credentials(self) -> bool:
        """Initialize Google Cloud credentials"""
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
            else:
                self.credentials, _ = default()
            
            logger.info("Google Cloud credentials initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud credentials: {str(e)}")
            return False
    
    def get_access_token(self) -> Optional[str]:
        """Get current access token"""
        try:
            if self.credentials:
                self.credentials.refresh(Request())
                return self.credentials.token
            return None
        except Exception as e:
            logger.error(f"Failed to get access token: {str(e)}")
            return None
    
    def validate_service_account(self) -> Dict[str, Any]:
        """Validate service account permissions"""
        try:
            if not self.credentials:
                return {'valid': False, 'error': 'No credentials available'}
            
            # Refresh credentials to validate
            self.credentials.refresh(Request())
            
            return {
                'valid': True,
                'service_account_email': getattr(self.credentials, 'service_account_email', 'unknown'),
                'project_id': self.project_id,
                'token_expiry': getattr(self.credentials, 'expiry', None)
            }
            
        except Exception as e:
            logger.error(f"Service account validation failed: {str(e)}")
            return {'valid': False, 'error': str(e)}

class SSLManager:
    """SSL/TLS certificate management"""
    
    def __init__(self):
        self.cert_path = config.security.ssl_cert_path
        self.key_path = config.security.ssl_key_path
    
    def validate_certificate(self) -> Dict[str, Any]:
        """Validate SSL certificate"""
        try:
            if not os.path.exists(self.cert_path) or not os.path.exists(self.key_path):
                return {
                    'valid': False,
                    'error': 'Certificate or key file not found'
                }
            
            # Load and validate certificate
            with open(self.cert_path, 'r') as cert_file:
                cert_data = cert_file.read()
            
            # Basic validation - check if it looks like a certificate
            if '-----BEGIN CERTIFICATE-----' in cert_data and '-----END CERTIFICATE-----' in cert_data:
                return {
                    'valid': True,
                    'cert_path': self.cert_path,
                    'key_path': self.key_path,
                    'message': 'Certificate files found and appear valid'
                }
            else:
                return {
                    'valid': False,
                    'error': 'Certificate file format appears invalid'
                }
                
        except Exception as e:
            logger.error(f"Certificate validation failed: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def get_ssl_context(self):
        """Get SSL context for Flask app"""
        try:
            if os.path.exists(self.cert_path) and os.path.exists(self.key_path):
                return ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            return None
        except Exception as e:
            logger.error(f"Failed to create SSL context: {str(e)}")
            return None

class SecurityMiddleware:
    """Security middleware for API endpoints"""
    
    def __init__(self):
        self.jwt_manager = JWTManager()
        self.encryption_manager = EncryptionManager()
        self.rate_limits = {}  # Simple rate limiting storage
    
    def require_auth(self, f):
        """Decorator to require authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                return jsonify({'error': 'Authorization header required'}), 401
            
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
                verification = self.jwt_manager.verify_token(token)
                
                if not verification['valid']:
                    return jsonify({'error': verification['error']}), 401
                
                # Add user data to request context
                request.user_data = verification['user_data']
                
            except (IndexError, AttributeError):
                return jsonify({'error': 'Invalid authorization header format'}), 401
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    def rate_limit(self, requests_per_minute: int = 60):
        """Decorator for rate limiting"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                client_id = request.remote_addr
                current_time = datetime.utcnow()
                
                # Clean old entries
                cutoff_time = current_time - timedelta(minutes=1)
                if client_id in self.rate_limits:
                    self.rate_limits[client_id] = [
                        timestamp for timestamp in self.rate_limits[client_id]
                        if timestamp > cutoff_time
                    ]
                else:
                    self.rate_limits[client_id] = []
                
                # Check rate limit
                if len(self.rate_limits[client_id]) >= requests_per_minute:
                    return jsonify({'error': 'Rate limit exceeded'}), 429
                
                # Add current request
                self.rate_limits[client_id].append(current_time)
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key (simple implementation)"""
        # In production, this should check against a database or external service
        valid_keys = os.getenv('VALID_API_KEYS', '').split(',')
        return api_key in valid_keys
    
    def secure_headers(self, response):
        """Add security headers to response"""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        return response

class SecurityManager:
    """Comprehensive security manager"""
    
    def __init__(self):
        self.encryption = EncryptionManager()
        self.jwt = JWTManager()
        self.iam = GoogleIAMManager()
        self.ssl = SSLManager()
        self.middleware = SecurityMiddleware()
    
    def perform_security_audit(self) -> Dict[str, Any]:
        """Perform comprehensive security audit"""
        audit_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        # Check encryption
        try:
            test_data = "test_encryption"
            encrypted = self.encryption.encrypt_data(test_data)
            decrypted = self.encryption.decrypt_data(encrypted)
            audit_results['checks']['encryption'] = {
                'status': 'passed' if decrypted == test_data else 'failed',
                'message': 'Encryption/decryption working properly'
            }
        except Exception as e:
            audit_results['checks']['encryption'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Check JWT
        try:
            token_result = self.jwt.generate_token({'test': 'user'})
            if token_result['success']:
                verify_result = self.jwt.verify_token(token_result['token'])
                audit_results['checks']['jwt'] = {
                    'status': 'passed' if verify_result['valid'] else 'failed',
                    'message': 'JWT generation and verification working'
                }
            else:
                audit_results['checks']['jwt'] = {
                    'status': 'failed',
                    'error': token_result['error']
                }
        except Exception as e:
            audit_results['checks']['jwt'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Check Google IAM
        iam_validation = self.iam.validate_service_account()
        audit_results['checks']['google_iam'] = {
            'status': 'passed' if iam_validation['valid'] else 'warning',
            'details': iam_validation
        }
        
        # Check SSL certificates
        ssl_validation = self.ssl.validate_certificate()
        audit_results['checks']['ssl'] = {
            'status': 'passed' if ssl_validation['valid'] else 'warning',
            'details': ssl_validation
        }
        
        # Overall status
        failed_checks = [check for check, result in audit_results['checks'].items() 
                        if result.get('status') == 'failed']
        
        if failed_checks:
            audit_results['overall_status'] = 'failed'
            audit_results['failed_checks'] = failed_checks
        else:
            audit_results['overall_status'] = 'passed'
        
        return audit_results
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get recommended security headers"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }

# Initialize security manager
security_manager = SecurityManager()
