"""
Google Cloud Storage integration for document storage and management
"""
from google.cloud import storage
from google.oauth2 import service_account
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import mimetypes
from config import config

logger = logging.getLogger(__name__)

class GoogleCloudStorageManager:
    """Google Cloud Storage manager for document storage"""
    
    def __init__(self):
        self.project_id = config.cloud.google_project_id
        self.bucket_name = config.cloud.google_storage_bucket
        self.credentials_path = config.cloud.google_credentials_path
        self.client = None
        self.bucket = None
        
        # Initialize client
        self.initialize_client()
    
    def initialize_client(self) -> bool:
        """Initialize Google Cloud Storage client"""
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.client = storage.Client(
                    project=self.project_id,
                    credentials=credentials
                )
            else:
                # Try default credentials
                self.client = storage.Client(project=self.project_id)
            
            # Get or create bucket
            if self.bucket_name:
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"Initialized Google Cloud Storage client for bucket: {self.bucket_name}")
                return True
            else:
                logger.error("Google Cloud Storage bucket name not configured")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Storage client: {str(e)}")
            return False
    
    def upload_file(self, file_path: str, destination_path: str = None, metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """Upload file to Google Cloud Storage"""
        if not self.client or not self.bucket:
            return {'success': False, 'error': 'Storage client not initialized'}
        
        try:
            # Use filename as destination if not provided
            if not destination_path:
                destination_path = os.path.basename(file_path)
            
            # Create blob
            blob = self.bucket.blob(destination_path)
            
            # Set metadata
            if metadata:
                blob.metadata = metadata
            
            # Detect content type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type:
                blob.content_type = content_type
            
            # Upload file
            with open(file_path, 'rb') as file_obj:
                blob.upload_from_file(file_obj)
            
            logger.info(f"Successfully uploaded {file_path} to {destination_path}")
            
            return {
                'success': True,
                'bucket': self.bucket_name,
                'blob_name': destination_path,
                'public_url': blob.public_url,
                'size': blob.size,
                'updated': blob.updated.isoformat() if blob.updated else None,
                'content_type': blob.content_type
            }
            
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def download_file(self, blob_name: str, local_path: str) -> Dict[str, Any]:
        """Download file from Google Cloud Storage"""
        if not self.client or not self.bucket:
            return {'success': False, 'error': 'Storage client not initialized'}
        
        try:
            blob = self.bucket.blob(blob_name)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            blob.download_to_filename(local_path)
            
            logger.info(f"Successfully downloaded {blob_name} to {local_path}")
            
            return {
                'success': True,
                'local_path': local_path,
                'blob_name': blob_name,
                'size': os.path.getsize(local_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to download file {blob_name}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def list_files(self, prefix: str = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """List files in Google Cloud Storage bucket"""
        if not self.client or not self.bucket:
            return []
        
        try:
            blobs = self.client.list_blobs(
                self.bucket,
                prefix=prefix,
                max_results=max_results
            )
            
            files = []
            for blob in blobs:
                files.append({
                    'name': blob.name,
                    'size': blob.size,
                    'content_type': blob.content_type,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'public_url': blob.public_url,
                    'metadata': blob.metadata
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            return []
    
    def delete_file(self, blob_name: str) -> Dict[str, Any]:
        """Delete file from Google Cloud Storage"""
        if not self.client or not self.bucket:
            return {'success': False, 'error': 'Storage client not initialized'}
        
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            
            logger.info(f"Successfully deleted {blob_name}")
            
            return {
                'success': True,
                'blob_name': blob_name,
                'deleted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to delete file {blob_name}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def generate_signed_url(self, blob_name: str, expiration_minutes: int = 60, method: str = 'GET') -> Dict[str, Any]:
        """Generate signed URL for secure access to file"""
        if not self.client or not self.bucket:
            return {'success': False, 'error': 'Storage client not initialized'}
        
        try:
            blob = self.bucket.blob(blob_name)
            
            # Generate signed URL
            url = blob.generate_signed_url(
                expiration=datetime.utcnow() + timedelta(minutes=expiration_minutes),
                method=method
            )
            
            return {
                'success': True,
                'signed_url': url,
                'expires_at': (datetime.utcnow() + timedelta(minutes=expiration_minutes)).isoformat(),
                'blob_name': blob_name,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {blob_name}: {str(e)}")
            return {'success': False, 'error': str(e)}

class DocumentStorageManager:
    """High-level document storage manager"""
    
    def __init__(self):
        self.gcs = GoogleCloudStorageManager()
        self.storage_prefix = "documents/"
        self.processed_prefix = "processed/"
        
    def store_uploaded_document(self, file_path: str, document_id: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store uploaded document with proper organization"""
        try:
            # Prepare metadata
            doc_metadata = {
                'document_id': document_id,
                'upload_timestamp': datetime.now().isoformat(),
                'original_filename': os.path.basename(file_path)
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            # Create storage path
            file_extension = os.path.splitext(file_path)[1]
            storage_path = f"{self.storage_prefix}{document_id}{file_extension}"
            
            # Upload to Google Cloud Storage
            result = self.gcs.upload_file(file_path, storage_path, doc_metadata)
            
            if result['success']:
                result['storage_path'] = storage_path
                result['document_id'] = document_id
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to store document {document_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def store_processed_results(self, document_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Store processing results as JSON"""
        try:
            # Create results file path
            results_filename = f"{document_id}_results.json"
            results_path = f"{self.processed_prefix}{results_filename}"
            
            # Create temporary file
            temp_file = f"temp/{results_filename}"
            os.makedirs("temp", exist_ok=True)
            
            # Write results to temp file
            with open(temp_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Upload to storage
            metadata = {
                'document_id': document_id,
                'content_type': 'processing_results',
                'created_at': datetime.now().isoformat()
            }
            
            result = self.gcs.upload_file(temp_file, results_path, metadata)
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            if result['success']:
                result['results_path'] = results_path
                result['document_id'] = document_id
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to store processing results for {document_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get information about stored document"""
        try:
            # List files with document ID prefix
            files = self.gcs.list_files(prefix=f"{self.storage_prefix}{document_id}")
            
            document_info = {
                'document_id': document_id,
                'files': files,
                'found': len(files) > 0
            }
            
            # Look for processing results
            results_files = self.gcs.list_files(prefix=f"{self.processed_prefix}{document_id}")
            if results_files:
                document_info['processing_results'] = results_files
            
            return document_info
            
        except Exception as e:
            logger.error(f"Failed to get document info for {document_id}: {str(e)}")
            return {'document_id': document_id, 'error': str(e)}
    
    def cleanup_old_files(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up files older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            files_to_delete = []
            
            # List all files
            all_files = self.gcs.list_files()
            
            for file_info in all_files:
                if file_info.get('created'):
                    created_date = datetime.fromisoformat(file_info['created'].replace('Z', '+00:00'))
                    if created_date < cutoff_date:
                        files_to_delete.append(file_info['name'])
            
            # Delete old files
            deleted_files = []
            for file_name in files_to_delete:
                result = self.gcs.delete_file(file_name)
                if result['success']:
                    deleted_files.append(file_name)
            
            return {
                'success': True,
                'deleted_count': len(deleted_files),
                'deleted_files': deleted_files,
                'cutoff_date': cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {str(e)}")
            return {'success': False, 'error': str(e)}

# Initialize storage managers
storage_manager = DocumentStorageManager()