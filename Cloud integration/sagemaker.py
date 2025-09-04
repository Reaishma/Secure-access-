"""
AWS SageMaker integration for machine learning operations
"""
import boto3
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from config import config

logger = logging.getLogger(__name__)

class SageMakerManager:
    """AWS SageMaker manager for ML operations"""
    
    def __init__(self):
        self.aws_access_key_id = config.cloud.aws_access_key_id
        self.aws_secret_access_key = config.cloud.aws_secret_access_key
        self.region = config.cloud.aws_region
        self.sagemaker_role = config.cloud.sagemaker_role
        
        # Initialize clients
        self.sagemaker_client = None
        self.runtime_client = None
        self.s3_client = None
        
        self.initialize_clients()
    
    def initialize_clients(self) -> bool:
        """Initialize AWS SageMaker clients"""
        try:
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region
            )
            
            self.sagemaker_client = session.client('sagemaker')
            self.runtime_client = session.client('sagemaker-runtime')
            self.s3_client = session.client('s3')
            
            logger.info(f"Initialized AWS SageMaker clients for region: {self.region}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS SageMaker clients: {str(e)}")
            return False
    
    def create_training_job(self, job_name: str, algorithm_specification: Dict[str, Any], 
                           input_data_config: List[Dict[str, Any]], output_data_config: Dict[str, Any],
                           instance_type: str = "ml.m5.large", instance_count: int = 1) -> Dict[str, Any]:
        """Create SageMaker training job"""
        try:
            response = self.sagemaker_client.create_training_job(
                TrainingJobName=job_name,
                AlgorithmSpecification=algorithm_specification,
                RoleArn=self.sagemaker_role,
                InputDataConfig=input_data_config,
                OutputDataConfig=output_data_config,
                ResourceConfig={
                    'InstanceType': instance_type,
                    'InstanceCount': instance_count,
                    'VolumeSizeInGB': 30
                },
                StoppingCondition={
                    'MaxRuntimeInSeconds': 86400  # 24 hours
                }
            )
            
            logger.info(f"Created SageMaker training job: {job_name}")
            
            return {
                'success': True,
                'job_name': job_name,
                'job_arn': response['TrainingJobArn']
            }
            
        except Exception as e:
            logger.error(f"Failed to create SageMaker training job: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_model(self, model_name: str, model_data_url: str, image_uri: str) -> Dict[str, Any]:
        """Create SageMaker model"""
        try:
            response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': model_data_url
                },
                ExecutionRoleArn=self.sagemaker_role
            )
            
            logger.info(f"Created SageMaker model: {model_name}")
            
            return {
                'success': True,
                'model_name': model_name,
                'model_arn': response['ModelArn']
            }
            
        except Exception as e:
            logger.error(f"Failed to create SageMaker model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_endpoint_config(self, config_name: str, model_name: str, 
                              instance_type: str = "ml.m5.large", instance_count: int = 1) -> Dict[str, Any]:
        """Create SageMaker endpoint configuration"""
        try:
            response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': instance_count,
                        'InstanceType': instance_type
                    }
                ]
            )
            
            logger.info(f"Created SageMaker endpoint config: {config_name}")
            
            return {
                'success': True,
                'config_name': config_name,
                'config_arn': response['EndpointConfigArn']
            }
            
        except Exception as e:
            logger.error(f"Failed to create SageMaker endpoint config: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_endpoint(self, endpoint_name: str, config_name: str) -> Dict[str, Any]:
        """Create SageMaker endpoint"""
        try:
            response = self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            
            logger.info(f"Created SageMaker endpoint: {endpoint_name}")
            
            return {
                'success': True,
                'endpoint_name': endpoint_name,
                'endpoint_arn': response['EndpointArn']
            }
            
        except Exception as e:
            logger.error(f"Failed to create SageMaker endpoint: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def deploy_model(self, model_name: str, endpoint_name: str, instance_type: str = "ml.m5.large") -> Dict[str, Any]:
        """Deploy model to SageMaker endpoint"""
        try:
            # Create endpoint config
            config_name = f"{model_name}-config-{int(datetime.now().timestamp())}"
            config_result = self.create_endpoint_config(config_name, model_name, instance_type)
            
            if not config_result['success']:
                return config_result
            
            # Create endpoint
            endpoint_result = self.create_endpoint(endpoint_name, config_name)
            
            if not endpoint_result['success']:
                return endpoint_result
            
            # Wait for endpoint to be in service
            waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=endpoint_name)
            
            return {
                'success': True,
                'model_name': model_name,
                'endpoint_name': endpoint_name,
                'config_name': config_name,
                'status': 'deployed'
            }
            
        except Exception as e:
            logger.error(f"Failed to deploy SageMaker model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, endpoint_name: str, data: bytes, content_type: str = "application/json") -> Dict[str, Any]:
        """Make predictions using SageMaker endpoint"""
        try:
            response = self.runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=data
            )
            
            result = response['Body'].read()
            
            # Try to parse as JSON
            try:
                predictions = json.loads(result.decode())
            except:
                predictions = result.decode()
            
            return {
                'success': True,
                'predictions': predictions,
                'endpoint_name': endpoint_name
            }
            
        except Exception as e:
            logger.error(f"SageMaker prediction failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def batch_transform(self, job_name: str, model_name: str, input_location: str, 
                       output_location: str, instance_type: str = "ml.m5.large") -> Dict[str, Any]:
        """Create batch transform job"""
        try:
            response = self.sagemaker_client.create_transform_job(
                TransformJobName=job_name,
                ModelName=model_name,
                TransformInput={
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': input_location
                        }
                    },
                    'ContentType': 'application/json'
                },
                TransformOutput={
                    'S3OutputPath': output_location
                },
                TransformResources={
                    'InstanceType': instance_type,
                    'InstanceCount': 1
                }
            )
            
            logger.info(f"Created SageMaker batch transform job: {job_name}")
            
            return {
                'success': True,
                'job_name': job_name,
                'job_arn': response['TransformJobArn']
            }
            
        except Exception as e:
            logger.error(f"Failed to create SageMaker batch transform job: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List SageMaker models"""
        try:
            response = self.sagemaker_client.list_models()
            
            models = []
            for model in response['Models']:
                models.append({
                    'name': model['ModelName'],
                    'arn': model['ModelArn'],
                    'creation_time': model['CreationTime'].isoformat()
                })
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list SageMaker models: {str(e)}")
            return []
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List SageMaker endpoints"""
        try:
            response = self.sagemaker_client.list_endpoints()
            
            endpoints = []
            for endpoint in response['Endpoints']:
                endpoints.append({
                    'name': endpoint['EndpointName'],
                    'arn': endpoint['EndpointArn'],
                    'status': endpoint['EndpointStatus'],
                    'creation_time': endpoint['CreationTime'].isoformat()
                })
            
            return endpoints
            
        except Exception as e:
            logger.error(f"Failed to list SageMaker endpoints: {str(e)}")
            return []
    
    def get_training_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get training job status"""
        try:
            response = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
            
            return {
                'success': True,
                'job_name': job_name,
                'status': response['TrainingJobStatus'],
                'creation_time': response['CreationTime'].isoformat(),
                'training_start_time': response.get('TrainingStartTime', {}).isoformat() if response.get('TrainingStartTime') else None,
                'training_end_time': response.get('TrainingEndTime', {}).isoformat() if response.get('TrainingEndTime') else None,
                'model_artifacts': response.get('ModelArtifacts', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get training job status: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def delete_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Delete SageMaker endpoint"""
        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            logger.info(f"Deleted SageMaker endpoint: {endpoint_name}")
            
            return {
                'success': True,
                'endpoint_name': endpoint_name,
                'deleted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to delete SageMaker endpoint: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete SageMaker model"""
        try:
            self.sagemaker_client.delete_model(ModelName=model_name)
            
            logger.info(f"Deleted SageMaker model: {model_name}")
            
            return {
                'success': True,
                'model_name': model_name,
                'deleted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to delete SageMaker model: {str(e)}")
            return {'success': False, 'error': str(e)}

class SageMakerBuiltinAlgorithms:
    """SageMaker built-in algorithms wrapper"""
    
    def __init__(self, sagemaker_manager: SageMakerManager):
        self.sagemaker = sagemaker_manager
        self.region = sagemaker_manager.region
    
    def get_algorithm_image_uri(self, algorithm_name: str, version: str = "latest") -> str:
        """Get algorithm image URI"""
        # Mapping of algorithms to their image URIs
        algorithm_images = {
            "xgboost": f"246618743249.dkr.ecr.{self.region}.amazonaws.com/xgboost:{version}",
            "linear-learner": f"174872318107.dkr.ecr.{self.region}.amazonaws.com/linear-learner:{version}",
            "blazingtext": f"811284229777.dkr.ecr.{self.region}.amazonaws.com/blazingtext:{version}",
            "image-classification": f"811284229777.dkr.ecr.{self.region}.amazonaws.com/image-classification:{version}"
        }
        
        return algorithm_images.get(algorithm_name, "")
    
    def train_text_classifier(self, job_name: str, training_data_s3: str, validation_data_s3: str,
                             output_s3: str, instance_type: str = "ml.m5.large") -> Dict[str, Any]:
        """Train text classifier using BlazingText"""
        try:
            algorithm_spec = {
                'TrainingImage': self.get_algorithm_image_uri('blazingtext'),
                'TrainingInputMode': 'File'
            }
            
            input_config = [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': training_data_s3,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/plain'
                }
            ]
            
            if validation_data_s3:
                input_config.append({
                    'ChannelName': 'validation',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': validation_data_s3,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/plain'
                })
            
            output_config = {
                'S3OutputPath': output_s3
            }
            
            return self.sagemaker.create_training_job(
                job_name=job_name,
                algorithm_specification=algorithm_spec,
                input_data_config=input_config,
                output_data_config=output_config,
                instance_type=instance_type
            )
            
        except Exception as e:
            logger.error(f"Failed to train text classifier: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def train_xgboost_classifier(self, job_name: str, training_data_s3: str, validation_data_s3: str,
                                output_s3: str, hyperparameters: Dict[str, str] = None) -> Dict[str, Any]:
        """Train XGBoost classifier"""
        try:
            default_hyperparameters = {
                'objective': 'multi:softmax',
                'num_class': '5',
                'num_round': '100'
            }
            
            if hyperparameters:
                default_hyperparameters.update(hyperparameters)
            
            algorithm_spec = {
                'TrainingImage': self.get_algorithm_image_uri('xgboost'),
                'TrainingInputMode': 'File'
            }
            
            input_config = [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': training_data_s3,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'csv'
                },
                {
                    'ChannelName': 'validation',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': validation_data_s3,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'csv'
                }
            ]
            
            output_config = {
                'S3OutputPath': output_s3
            }
            
            # Create training job with hyperparameters
            response = self.sagemaker.sagemaker_client.create_training_job(
                TrainingJobName=job_name,
                AlgorithmSpecification=algorithm_spec,
                RoleArn=self.sagemaker.sagemaker_role,
                InputDataConfig=input_config,
                OutputDataConfig=output_config,
                ResourceConfig={
                    'InstanceType': 'ml.m5.large',
                    'InstanceCount': 1,
                    'VolumeSizeInGB': 30
                },
                StoppingCondition={
                    'MaxRuntimeInSeconds': 86400
                },
                HyperParameters=default_hyperparameters
            )
            
            return {
                'success': True,
                'job_name': job_name,
                'job_arn': response['TrainingJobArn']
            }
            
        except Exception as e:
            logger.error(f"Failed to train XGBoost classifier: {str(e)}")
            return {'success': False, 'error': str(e)}

class SageMakerPipeline:
    """SageMaker pipeline management for document processing"""
    
    def __init__(self):
        self.sagemaker = SageMakerManager()
        self.algorithms = SageMakerBuiltinAlgorithms(self.sagemaker)
    
    def create_document_processing_pipeline(self, pipeline_name: str, s3_bucket: str) -> Dict[str, Any]:
        """Create end-to-end document processing pipeline"""
        try:
            # Step 1: Data preprocessing
            preprocess_job_name = f"{pipeline_name}-preprocess-{int(datetime.now().timestamp())}"
            
            # Step 2: Model training
            training_job_name = f"{pipeline_name}-training-{int(datetime.now().timestamp())}"
            
            # Step 3: Model evaluation
            evaluation_job_name = f"{pipeline_name}-evaluation-{int(datetime.now().timestamp())}"
            
            # Create training job for document classification
            training_result = self.algorithms.train_text_classifier(
                job_name=training_job_name,
                training_data_s3=f"s3://{s3_bucket}/data/training/",
                validation_data_s3=f"s3://{s3_bucket}/data/validation/",
                output_s3=f"s3://{s3_bucket}/models/"
            )
            
            if training_result['success']:
                # Create model after training completes
                model_name = f"{pipeline_name}-model"
                model_data_url = f"s3://{s3_bucket}/models/{training_job_name}/output/model.tar.gz"
                image_uri = self.algorithms.get_algorithm_image_uri('blazingtext')
                
                return {
                    'success': True,
                    'pipeline_name': pipeline_name,
                    'training_job': training_result,
                    'model_name': model_name,
                    'model_data_url': model_data_url
                }
            else:
                return training_result
                
        except Exception as e:
            logger.error(f"Failed to create SageMaker pipeline: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall SageMaker pipeline status"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'region': self.sagemaker.region,
                'models': len(self.sagemaker.list_models()),
                'endpoints': len(self.sagemaker.list_endpoints()),
                'sagemaker_role': self.sagemaker.sagemaker_role,
                'status': 'active' if self.sagemaker.sagemaker_client else 'inactive'
            }
            
        except Exception as e:
            logger.error(f"Failed to get SageMaker status: {str(e)}")
            return {'error': str(e)}

# Initialize SageMaker manager
sagemaker_manager = SageMakerPipeline()