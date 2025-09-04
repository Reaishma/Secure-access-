"""
Google Cloud AI Platform integration for machine learning operations
"""
from google.cloud import aiplatform
from google.oauth2 import service_account
import os
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from config import config

logger = logging.getLogger(__name__)

class GoogleAIPlatformManager:
    """Google Cloud AI Platform manager for ML operations"""
    
    def __init__(self):
        self.project_id = config.cloud.google_project_id
        self.region = config.cloud.aws_region  # Using same region config
        self.credentials_path = config.cloud.google_credentials_path
        self.credentials = None
        
        # Initialize client
        self.initialize_client()
    
    def initialize_client(self) -> bool:
        """Initialize Google AI Platform client"""
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
            
            # Initialize AI Platform
            aiplatform.init(
                project=self.project_id,
                location=self.region or 'us-central1',
                credentials=self.credentials
            )
            
            logger.info(f"Initialized Google AI Platform client for project: {self.project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Google AI Platform client: {str(e)}")
            return False
    
    def create_dataset(self, display_name: str, metadata_schema_uri: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a dataset on AI Platform"""
        try:
            dataset = aiplatform.Dataset.create(
                display_name=display_name,
                metadata_schema_uri=metadata_schema_uri,
                metadata=metadata
            )
            
            logger.info(f"Created dataset: {dataset.display_name}")
            
            return {
                'success': True,
                'dataset_id': dataset.name,
                'display_name': dataset.display_name,
                'create_time': dataset.create_time.isoformat() if dataset.create_time else None
            }
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def upload_model(self, model_path: str, display_name: str, serving_container_image_uri: str) -> Dict[str, Any]:
        """Upload model to AI Platform"""
        try:
            # Upload model
            model = aiplatform.Model.upload(
                display_name=display_name,
                artifact_uri=model_path,
                serving_container_image_uri=serving_container_image_uri
            )
            
            logger.info(f"Uploaded model: {model.display_name}")
            
            return {
                'success': True,
                'model_id': model.name,
                'display_name': model.display_name,
                'model_url': model.uri
            }
            
        except Exception as e:
            logger.error(f"Failed to upload model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_endpoint(self, display_name: str) -> Dict[str, Any]:
        """Create endpoint for model serving"""
        try:
            endpoint = aiplatform.Endpoint.create(display_name=display_name)
            
            logger.info(f"Created endpoint: {endpoint.display_name}")
            
            return {
                'success': True,
                'endpoint_id': endpoint.name,
                'display_name': endpoint.display_name
            }
            
        except Exception as e:
            logger.error(f"Failed to create endpoint: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def deploy_model(self, model_id: str, endpoint_id: str, machine_type: str = "n1-standard-4") -> Dict[str, Any]:
        """Deploy model to endpoint"""
        try:
            model = aiplatform.Model(model_id)
            endpoint = aiplatform.Endpoint(endpoint_id)
            
            # Deploy model to endpoint
            endpoint.deploy(
                model=model,
                deployed_model_display_name=f"{model.display_name}_deployment",
                machine_type=machine_type
            )
            
            logger.info(f"Deployed model {model_id} to endpoint {endpoint_id}")
            
            return {
                'success': True,
                'model_id': model_id,
                'endpoint_id': endpoint_id,
                'deployment_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, endpoint_id: str, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make predictions using deployed model"""
        try:
            endpoint = aiplatform.Endpoint(endpoint_id)
            
            # Make prediction
            prediction = endpoint.predict(instances=instances)
            
            return {
                'success': True,
                'predictions': prediction.predictions,
                'deployed_model_id': prediction.deployed_model_id,
                'model_version_id': prediction.model_version_id
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def batch_predict(self, job_display_name: str, model_name: str, gcs_source_uris: List[str], 
                     gcs_destination_output_uri_prefix: str) -> Dict[str, Any]:
        """Run batch prediction job"""
        try:
            model = aiplatform.Model(model_name)
            
            # Create batch prediction job
            batch_predict_job = model.batch_predict(
                job_display_name=job_display_name,
                gcs_source=gcs_source_uris,
                gcs_destination_prefix=gcs_destination_output_uri_prefix,
                instances_format='jsonl',
                predictions_format='jsonl'
            )
            
            logger.info(f"Started batch prediction job: {job_display_name}")
            
            return {
                'success': True,
                'job_id': batch_predict_job.name,
                'job_display_name': job_display_name,
                'state': batch_predict_job.state.name
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_training_job(self, display_name: str, script_path: str, container_uri: str, 
                           requirements: List[str] = None) -> Dict[str, Any]:
        """Create custom training job"""
        try:
            # Define training job
            job = aiplatform.CustomJob(
                display_name=display_name,
                script_path=script_path,
                container_uri=container_uri,
                requirements=requirements or []
            )
            
            # Submit training job
            job.run()
            
            logger.info(f"Started training job: {display_name}")
            
            return {
                'success': True,
                'job_id': job.name,
                'display_name': display_name,
                'state': job.state.name
            }
            
        except Exception as e:
            logger.error(f"Training job creation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def list_models(self, filter_expr: str = None) -> List[Dict[str, Any]]:
        """List models in AI Platform"""
        try:
            models = aiplatform.Model.list(filter=filter_expr)
            
            model_list = []
            for model in models:
                model_list.append({
                    'name': model.name,
                    'display_name': model.display_name,
                    'create_time': model.create_time.isoformat() if model.create_time else None,
                    'update_time': model.update_time.isoformat() if model.update_time else None,
                    'labels': model.labels
                })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
    
    def list_endpoints(self, filter_expr: str = None) -> List[Dict[str, Any]]:
        """List endpoints in AI Platform"""
        try:
            endpoints = aiplatform.Endpoint.list(filter=filter_expr)
            
            endpoint_list = []
            for endpoint in endpoints:
                endpoint_list.append({
                    'name': endpoint.name,
                    'display_name': endpoint.display_name,
                    'create_time': endpoint.create_time.isoformat() if endpoint.create_time else None,
                    'update_time': endpoint.update_time.isoformat() if endpoint.update_time else None
                })
            
            return endpoint_list
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {str(e)}")
            return []
    
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete model from AI Platform"""
        try:
            model = aiplatform.Model(model_id)
            model.delete()
            
            logger.info(f"Deleted model: {model_id}")
            
            return {
                'success': True,
                'model_id': model_id,
                'deleted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def delete_endpoint(self, endpoint_id: str, force: bool = False) -> Dict[str, Any]:
        """Delete endpoint from AI Platform"""
        try:
            endpoint = aiplatform.Endpoint(endpoint_id)
            endpoint.delete(force=force)
            
            logger.info(f"Deleted endpoint: {endpoint_id}")
            
            return {
                'success': True,
                'endpoint_id': endpoint_id,
                'deleted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {str(e)}")
            return {'success': False, 'error': str(e)}

class AutoMLManager:
    """Google Cloud AutoML manager for automated ML"""
    
    def __init__(self):
        self.platform_manager = GoogleAIPlatformManager()
        self.project_id = config.cloud.google_project_id
    
    def create_text_classification_dataset(self, display_name: str, gcs_source_uri: str) -> Dict[str, Any]:
        """Create AutoML text classification dataset"""
        try:
            metadata_schema_uri = aiplatform.schema.dataset.metadata.text_classification
            
            dataset = aiplatform.TextDataset.create(
                display_name=display_name,
                gcs_source=gcs_source_uri,
                import_schema_uri=aiplatform.schema.dataset.ioformat.text.single_label_classification
            )
            
            return {
                'success': True,
                'dataset_id': dataset.name,
                'display_name': dataset.display_name
            }
            
        except Exception as e:
            logger.error(f"Failed to create AutoML text dataset: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def train_automl_model(self, dataset_id: str, display_name: str, target_column: str) -> Dict[str, Any]:
        """Train AutoML text classification model"""
        try:
            dataset = aiplatform.TextDataset(dataset_id)
            
            # Train AutoML model
            job = aiplatform.AutoMLTextTrainingJob(
                display_name=f"{display_name}_training_job"
            )
            
            model = job.run(
                dataset=dataset,
                target_column=target_column
            )
            
            return {
                'success': True,
                'model_id': model.name,
                'training_job_id': job.name,
                'display_name': display_name
            }
            
        except Exception as e:
            logger.error(f"AutoML training failed: {str(e)}")
            return {'success': False, 'error': str(e)}

class DocumentAIManager:
    """Google Cloud Document AI integration"""
    
    def __init__(self):
        self.project_id = config.cloud.google_project_id
        self.location = config.cloud.aws_region or 'us'  # Using same region config
    
    def process_document(self, file_path: str, processor_id: str) -> Dict[str, Any]:
        """Process document using Document AI"""
        try:
            from google.cloud import documentai
            
            # Initialize client
            client = documentai.DocumentProcessorServiceClient()
            
            # Read document
            with open(file_path, "rb") as document:
                document_content = document.read()
            
            # Configure request
            name = client.processor_path(self.project_id, self.location, processor_id)
            
            request = documentai.ProcessRequest(
                name=name,
                raw_document=documentai.RawDocument(
                    content=document_content,
                    mime_type="application/pdf"
                ),
            )
            
            # Process document
            result = client.process_document(request=request)
            document = result.document
            
            # Extract information
            entities = []
            for entity in document.entities:
                entities.append({
                    'type': entity.type_,
                    'mention_text': entity.mention_text,
                    'confidence': entity.confidence
                })
            
            return {
                'success': True,
                'text': document.text,
                'entities': entities,
                'pages': len(document.pages)
            }
            
        except Exception as e:
            logger.error(f"Document AI processing failed: {str(e)}")
            return {'success': False, 'error': str(e)}

class CloudAIPlatformManager:
    """Comprehensive Google Cloud AI Platform manager"""
    
    def __init__(self):
        self.platform = GoogleAIPlatformManager()
        self.automl = AutoMLManager()
        self.document_ai = DocumentAIManager()
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get overall platform status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'project_id': self.platform.project_id,
            'region': self.platform.region,
            'models': len(self.platform.list_models()),
            'endpoints': len(self.platform.list_endpoints()),
            'status': 'active'
        }
    
    def deploy_document_classifier(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """Deploy document classification model"""
        try:
            # Upload model
            model_result = self.platform.upload_model(
                model_path=model_path,
                display_name=model_name,
                serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest"
            )
            
            if not model_result['success']:
                return model_result
            
            # Create endpoint
            endpoint_result = self.platform.create_endpoint(
                display_name=f"{model_name}_endpoint"
            )
            
            if not endpoint_result['success']:
                return endpoint_result
            
            # Deploy model to endpoint
            deploy_result = self.platform.deploy_model(
                model_id=model_result['model_id'],
                endpoint_id=endpoint_result['endpoint_id']
            )
            
            return {
                'success': deploy_result['success'],
                'model_id': model_result['model_id'],
                'endpoint_id': endpoint_result['endpoint_id'],
                'deployment_details': deploy_result
            }
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict_document_type(self, endpoint_id: str, text_content: str) -> Dict[str, Any]:
        """Predict document type using deployed model"""
        try:
            # Prepare instance for prediction
            instances = [{
                'text': text_content[:1000]  # Limit text length
            }]
            
            # Make prediction
            result = self.platform.predict(endpoint_id, instances)
            
            if result['success'] and result['predictions']:
                prediction = result['predictions'][0]
                
                # Process prediction results
                if isinstance(prediction, list) and len(prediction) > 0:
                    # Get class with highest probability
                    max_idx = np.argmax(prediction)
                    confidence = float(prediction[max_idx])
                    
                    # Map to document types (this would be based on your training labels)
                    document_types = ['invoice', 'contract', 'resume', 'report', 'email']
                    predicted_type = document_types[max_idx] if max_idx < len(document_types) else 'unknown'
                    
                    return {
                        'success': True,
                        'predicted_type': predicted_type,
                        'confidence': confidence,
                        'all_scores': {
                            document_types[i]: float(score) 
                            for i, score in enumerate(prediction) 
                            if i < len(document_types)
                        }
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Document type prediction failed: {str(e)}")
            return {'success': False, 'error': str(e)}

# Initialize Cloud AI Platform manager
cloud_ai_manager = CloudAIPlatformManager()