"""
Configuration management for Intelligent Document Processing System
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    # MySQL Configuration
    mysql_host: str = os.getenv('MYSQL_HOST', 'localhost')
    mysql_port: int = int(os.getenv('MYSQL_PORT', 3306))
    mysql_user: str = os.getenv('MYSQL_USER', 'root')
    mysql_password: str = os.getenv('MYSQL_PASSWORD', '')
    mysql_database: str = os.getenv('MYSQL_DATABASE', 'document_processing')
    
    # Cassandra Configuration
    cassandra_hosts: List[str] = field(default_factory=lambda: [os.getenv('CASSANDRA_HOST', '127.0.0.1')])
    cassandra_port: int = int(os.getenv('CASSANDRA_PORT', 9042))
    cassandra_keyspace: str = os.getenv('CASSANDRA_KEYSPACE', 'document_processing')

@dataclass
class CloudConfig:
    """Cloud services configuration"""
    # Google Cloud Configuration
    google_credentials_path: str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
    google_project_id: str = os.getenv('GOOGLE_PROJECT_ID', '')
    google_storage_bucket: str = os.getenv('GOOGLE_STORAGE_BUCKET', '')
    google_vision_api_key: str = os.getenv('GOOGLE_VISION_API_KEY', '')
    
    # AWS Configuration
    aws_access_key_id: str = os.getenv('AWS_ACCESS_KEY_ID', '')
    aws_secret_access_key: str = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    aws_region: str = os.getenv('AWS_REGION', 'us-east-1')
    sagemaker_role: str = os.getenv('SAGEMAKER_EXECUTION_ROLE', '')
    
    # Azure Configuration
    azure_tenant_id: str = os.getenv('AZURE_TENANT_ID', '')
    azure_client_id: str = os.getenv('AZURE_CLIENT_ID', '')
    azure_client_secret: str = os.getenv('AZURE_CLIENT_SECRET', '')
    azure_subscription_id: str = os.getenv('AZURE_SUBSCRIPTION_ID', '')
    azure_resource_group: str = os.getenv('AZURE_RESOURCE_GROUP', '')
    azure_workspace_name: str = os.getenv('AZURE_WORKSPACE_NAME', '')

@dataclass
class ModelConfig:
    """ML Model configuration"""
    # Model paths
    tensorflow_model_path: str = os.getenv('TENSORFLOW_MODEL_PATH', 'models/tensorflow/')
    pytorch_model_path: str = os.getenv('PYTORCH_MODEL_PATH', 'models/pytorch/')
    
    # Training parameters
    batch_size: int = int(os.getenv('BATCH_SIZE', 32))
    learning_rate: float = float(os.getenv('LEARNING_RATE', 0.001))
    epochs: int = int(os.getenv('EPOCHS', 100))
    
    # Prediction parameters
    confidence_threshold: float = float(os.getenv('CONFIDENCE_THRESHOLD', 0.8))

@dataclass
class OCRConfig:
    """OCR configuration"""
    tesseract_cmd: str = os.getenv('TESSERACT_CMD', 'tesseract')
    tesseract_config: str = '--psm 6'
    supported_languages: List[str] = field(default_factory=lambda: ['eng', 'spa', 'fra', 'deu'])
    
    # Google Vision API
    google_vision_features: List[str] = field(default_factory=lambda: [
        'TEXT_DETECTION',
        'DOCUMENT_TEXT_DETECTION',
        'LABEL_DETECTION'
    ])

@dataclass
class SecurityConfig:
    """Security configuration"""
    ssl_cert_path: str = os.getenv('SSL_CERT_PATH', '')
    ssl_key_path: str = os.getenv('SSL_KEY_PATH', '')
    encryption_key: str = os.getenv('ENCRYPTION_KEY', '')
    jwt_secret: str = os.getenv('JWT_SECRET', 'your-secret-key')
    session_timeout: int = int(os.getenv('SESSION_TIMEOUT', 3600))

@dataclass
class AppConfig:
    """Main application configuration"""
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    host: str = os.getenv('HOST', '0.0.0.0')
    port: int = int(os.getenv('PORT', 5000))
    max_file_size: int = int(os.getenv('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB
    upload_folder: str = os.getenv('UPLOAD_FOLDER', 'uploads/')
    processed_folder: str = os.getenv('PROCESSED_FOLDER', 'processed/')
    
    # Redis for caching
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', 6379))
    redis_db: int = int(os.getenv('REDIS_DB', 0))

class Config:
    """Centralized configuration manager"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.cloud = CloudConfig()
        self.model = ModelConfig()
        self.ocr = OCRConfig()
        self.security = SecurityConfig()
        self.app = AppConfig()
    
    def get_config(self, section: str) -> Any:
        """Get configuration section"""
        return getattr(self, section, None)
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation_results = {}
        
        # Check required environment variables
        required_vars = [
            'GOOGLE_PROJECT_ID',
            'GOOGLE_APPLICATION_CREDENTIALS',
            'MYSQL_HOST',
            'CASSANDRA_HOST'
        ]
        
        for var in required_vars:
            validation_results[var] = bool(os.getenv(var))
        
        return validation_results
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.app.upload_folder,
            self.app.processed_folder,
            self.model.tensorflow_model_path,
            self.model.pytorch_model_path,
            'logs/',
            'temp/'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config()