"""
Azure Machine Learning integration for cloud-based ML operations
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from config import config

logger = logging.getLogger(__name__)

class AzureMLManager:
    """Azure Machine Learning workspace manager"""
    
    def __init__(self):
        self.subscription_id = config.cloud.azure_subscription_id
        self.resource_group = config.cloud.azure_resource_group
        self.workspace_name = config.cloud.azure_workspace_name
        self.tenant_id = config.cloud.azure_tenant_id
        self.client_id = config.cloud.azure_client_id
        self.client_secret = config.cloud.azure_client_secret
        
        self.workspace = None
        self.ml_client = None
        
        # Initialize client
        self.initialize_client()
    
    def initialize_client(self) -> bool:
        """Initialize Azure ML client"""
        try:
            from azure.ai.ml import MLClient
            from azure.identity import ClientSecretCredential
            
            # Create credential
            credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            
            # Initialize ML client
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
            
            # Get workspace
            self.workspace = self.ml_client.workspaces.get(self.workspace_name)
            
            logger.info(f"Initialized Azure ML client for workspace: {self.workspace_name}")
            return True
            
        except ImportError:
            logger.warning("Azure ML SDK not available. Install azure-ai-ml package.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML client: {str(e)}")
            return False
    
    def create_dataset(self, name: str, description: str, local_path: str = None, 
                      datastore_path: str = None) -> Dict[str, Any]:
        """Create dataset in Azure ML"""
        try:
            from azure.ai.ml.entities import Data
            from azure.ai.ml.constants import AssetTypes
            
            if local_path and os.path.exists(local_path):
                # Create dataset from local file
                data_asset = Data(
                    name=name,
                    description=description,
                    path=local_path,
                    type=AssetTypes.URI_FILE
                )
            elif datastore_path:
                # Create dataset from datastore path
                data_asset = Data(
                    name=name,
                    description=description,
                    path=datastore_path,
                    type=AssetTypes.URI_FOLDER
                )
            else:
                return {'success': False, 'error': 'No valid path provided'}
            
            # Create dataset
            data_asset = self.ml_client.data.create_or_update(data_asset)
            
            logger.info(f"Created Azure ML dataset: {name}")
            
            return {
                'success': True,
                'name': data_asset.name,
                'version': data_asset.version,
                'id': data_asset.id
            }
            
        except Exception as e:
            logger.error(f"Failed to create Azure ML dataset: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def register_model(self, name: str, model_path: str, description: str = None) -> Dict[str, Any]:
        """Register model in Azure ML"""
        try:
            from azure.ai.ml.entities import Model
            from azure.ai.ml.constants import AssetTypes
            
            model = Model(
                name=name,
                path=model_path,
                description=description,
                type=AssetTypes.CUSTOM_MODEL
            )
            
            # Register model
            registered_model = self.ml_client.models.create_or_update(model)
            
            logger.info(f"Registered Azure ML model: {name}")
            
            return {
                'success': True,
                'name': registered_model.name,
                'version': registered_model.version,
                'id': registered_model.id
            }
            
        except Exception as e:
            logger.error(f"Failed to register Azure ML model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_compute_instance(self, name: str, vm_size: str = "Standard_DS3_v2") -> Dict[str, Any]:
        """Create compute instance for training"""
        try:
            from azure.ai.ml.entities import ComputeInstance
            
            compute_instance = ComputeInstance(
                name=name,
                size=vm_size
            )
            
            # Create compute instance
            compute_instance = self.ml_client.compute.begin_create_or_update(compute_instance)
            
            logger.info(f"Created Azure ML compute instance: {name}")
            
            return {
                'success': True,
                'name': compute_instance.name,
                'size': vm_size,
                'state': compute_instance.state
            }
            
        except Exception as e:
            logger.error(f"Failed to create Azure ML compute instance: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def submit_training_job(self, job_name: str, script_path: str, compute_target: str, 
                           environment_name: str = None) -> Dict[str, Any]:
        """Submit training job to Azure ML"""
        try:
            from azure.ai.ml import command
            
            # Create command job
            job = command(
                name=job_name,
                code=script_path,
                command="python main.py",
                compute=compute_target,
                environment=environment_name
            )
            
            # Submit job
            submitted_job = self.ml_client.jobs.create_or_update(job)
            
            logger.info(f"Submitted Azure ML training job: {job_name}")
            
            return {
                'success': True,
                'job_name': submitted_job.name,
                'status': submitted_job.status,
                'job_id': submitted_job.id
            }
            
        except Exception as e:
            logger.error(f"Failed to submit Azure ML training job: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def deploy_model(self, model_name: str, endpoint_name: str, deployment_name: str, 
                    vm_size: str = "Standard_DS3_v2", instance_count: int = 1) -> Dict[str, Any]:
        """Deploy model to Azure ML endpoint"""
        try:
            from azure.ai.ml.entities import (
                ManagedOnlineEndpoint,
                ManagedOnlineDeployment,
                Model
            )
            
            # Create endpoint
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description=f"Endpoint for {model_name}"
            )
            
            # Create endpoint
            endpoint = self.ml_client.online_endpoints.begin_create_or_update(endpoint)
            
            # Get model
            model = self.ml_client.models.get(model_name, version=None)
            
            # Create deployment
            deployment = ManagedOnlineDeployment(
                name=deployment_name,
                endpoint_name=endpoint_name,
                model=model,
                instance_type=vm_size,
                instance_count=instance_count
            )
            
            # Deploy model
            deployment = self.ml_client.online_deployments.begin_create_or_update(deployment)
            
            logger.info(f"Deployed Azure ML model: {model_name} to endpoint: {endpoint_name}")
            
            return {
                'success': True,
                'model_name': model_name,
                'endpoint_name': endpoint_name,
                'deployment_name': deployment_name
            }
            
        except Exception as e:
            logger.error(f"Failed to deploy Azure ML model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, endpoint_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make predictions using deployed model"""
        try:
            # Invoke endpoint
            predictions = self.ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                request_file=json.dumps(data)
            )
            
            return {
                'success': True,
                'predictions': json.loads(predictions),
                'endpoint_name': endpoint_name
            }
            
        except Exception as e:
            logger.error(f"Azure ML prediction failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List registered models"""
        try:
            models = self.ml_client.models.list()
            
            model_list = []
            for model in models:
                model_list.append({
                    'name': model.name,
                    'version': model.version,
                    'description': model.description,
                    'created_time': model.creation_context.created_at.isoformat() if model.creation_context else None
                })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Failed to list Azure ML models: {str(e)}")
            return []
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List online endpoints"""
        try:
            endpoints = self.ml_client.online_endpoints.list()
            
            endpoint_list = []
            for endpoint in endpoints:
                endpoint_list.append({
                    'name': endpoint.name,
                    'description': endpoint.description,
                    'provisioning_state': endpoint.provisioning_state,
                    'scoring_uri': endpoint.scoring_uri
                })
            
            return endpoint_list
            
        except Exception as e:
            logger.error(f"Failed to list Azure ML endpoints: {str(e)}")
            return []
    
    def delete_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Delete online endpoint"""
        try:
            self.ml_client.online_endpoints.begin_delete(endpoint_name)
            
            logger.info(f"Deleted Azure ML endpoint: {endpoint_name}")
            
            return {
                'success': True,
                'endpoint_name': endpoint_name,
                'deleted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to delete Azure ML endpoint: {str(e)}")
            return {'success': False, 'error': str(e)}

class AzureCognitiveServices:
    """Azure Cognitive Services integration"""
    
    def __init__(self):
        self.subscription_key = os.getenv('AZURE_COGNITIVE_SERVICES_KEY', '')
        self.endpoint = os.getenv('AZURE_COGNITIVE_SERVICES_ENDPOINT', '')
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text using Azure Text Analytics"""
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential
            
            if not self.subscription_key or not self.endpoint:
                return {'success': False, 'error': 'Azure Cognitive Services not configured'}
            
            # Initialize client
            client = TextAnalyticsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.subscription_key)
            )
            
            documents = [text]
            
            # Analyze sentiment
            sentiment_response = client.analyze_sentiment(documents=documents)
            sentiment_result = sentiment_response[0]
            
            # Extract key phrases
            key_phrases_response = client.extract_key_phrases(documents=documents)
            key_phrases_result = key_phrases_response[0]
            
            # Recognize entities
            entities_response = client.recognize_entities(documents=documents)
            entities_result = entities_response[0]
            
            return {
                'success': True,
                'sentiment': {
                    'sentiment': sentiment_result.sentiment,
                    'confidence_scores': {
                        'positive': sentiment_result.confidence_scores.positive,
                        'neutral': sentiment_result.confidence_scores.neutral,
                        'negative': sentiment_result.confidence_scores.negative
                    }
                },
                'key_phrases': key_phrases_result.key_phrases,
                'entities': [
                    {
                        'text': entity.text,
                        'category': entity.category,
                        'confidence_score': entity.confidence_score
                    }
                    for entity in entities_result.entities
                ]
            }
            
        except ImportError:
            logger.warning("Azure Cognitive Services SDK not available")
            return {'success': False, 'error': 'SDK not available'}
        except Exception as e:
            logger.error(f"Azure text analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def extract_form_data(self, document_path: str) -> Dict[str, Any]:
        """Extract form data using Azure Form Recognizer"""
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
            
            if not self.subscription_key or not self.endpoint:
                return {'success': False, 'error': 'Azure Form Recognizer not configured'}
            
            # Initialize client
            client = DocumentAnalysisClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.subscription_key)
            )
            
            # Analyze document
            with open(document_path, "rb") as document:
                poller = client.begin_analyze_document("prebuilt-document", document)
                result = poller.result()
            
            # Extract key-value pairs
            key_value_pairs = []
            for kv_pair in result.key_value_pairs:
                key_value_pairs.append({
                    'key': kv_pair.key.content if kv_pair.key else '',
                    'value': kv_pair.value.content if kv_pair.value else '',
                    'confidence': kv_pair.confidence
                })
            
            # Extract tables
            tables = []
            for table in result.tables:
                table_data = []
                for cell in table.cells:
                    table_data.append({
                        'content': cell.content,
                        'row_index': cell.row_index,
                        'column_index': cell.column_index,
                        'confidence': cell.confidence
                    })
                tables.append({
                    'row_count': table.row_count,
                    'column_count': table.column_count,
                    'cells': table_data
                })
            
            return {
                'success': True,
                'key_value_pairs': key_value_pairs,
                'tables': tables,
                'content': result.content
            }
            
        except ImportError:
            logger.warning("Azure Form Recognizer SDK not available")
            return {'success': False, 'error': 'SDK not available'}
        except Exception as e:
            logger.error(f"Azure form recognition failed: {str(e)}")
            return {'success': False, 'error': str(e)}

class AzureMLPipeline:
    """Azure ML Pipeline management"""
    
    def __init__(self):
        self.azure_ml = AzureMLManager()
        self.cognitive_services = AzureCognitiveServices()
    
    def create_document_processing_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """Create document processing pipeline"""
        try:
            from azure.ai.ml.dsl import pipeline
            from azure.ai.ml import Input, Output
            
            @pipeline(
                description="Document processing pipeline",
                default_compute="cpu-cluster"
            )
            def document_pipeline(input_data: Input):
                # Step 1: OCR processing
                ocr_step = self.create_ocr_step(input_data)
                
                # Step 2: Text analysis
                analysis_step = self.create_analysis_step(ocr_step.outputs.processed_text)
                
                # Step 3: Classification
                classification_step = self.create_classification_step(analysis_step.outputs.analyzed_text)
                
                return {
                    "ocr_results": ocr_step.outputs.processed_text,
                    "analysis_results": analysis_step.outputs.analyzed_text,
                    "classification_results": classification_step.outputs.classified_text
                }
            
            # Create pipeline
            pipeline_instance = document_pipeline()
            
            # Submit pipeline
            pipeline_job = self.azure_ml.ml_client.jobs.create_or_update(pipeline_instance)
            
            return {
                'success': True,
                'pipeline_name': pipeline_name,
                'pipeline_id': pipeline_job.id,
                'status': pipeline_job.status
            }
            
        except Exception as e:
            logger.error(f"Failed to create Azure ML pipeline: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_ocr_step(self, input_data):
        """Create OCR processing step"""
        # This would be implemented with actual Azure ML components
        pass
    
    def create_analysis_step(self, text_input):
        """Create text analysis step"""
        # This would be implemented with actual Azure ML components
        pass
    
    def create_classification_step(self, analyzed_text):
        """Create classification step"""
        # This would be implemented with actual Azure ML components
        pass
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall Azure ML pipeline status"""
        try:
            # Check workspace connection
            workspace_status = bool(self.azure_ml.workspace)
            
            # Get pipeline runs
            jobs = list(self.azure_ml.ml_client.jobs.list())
            
            return {
                'timestamp': datetime.now().isoformat(),
                'workspace_connected': workspace_status,
                'total_jobs': len(jobs),
                'active_jobs': len([job for job in jobs if job.status == 'Running']),
                'models': len(self.azure_ml.list_models()),
                'endpoints': len(self.azure_ml.list_endpoints())
            }
            
        except Exception as e:
            logger.error(f"Failed to get Azure ML status: {str(e)}")
            return {'error': str(e)}

# Initialize Azure ML manager
azure_ml_manager = AzureMLPipeline()