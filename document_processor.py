"""
Main document processor orchestrator that coordinates all processing steps
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile

# Import all processing modules
from config import config

logger = logging.getLogger(__name__)

# Initialize modules with error handling
ocr_engine = None
nlp_pipeline = None
entity_extractor = None
prediction_engine = None
database_manager = None
storage_manager = None
security_manager = None

try:
    from OCR import ocr_engine
    from NLP import nlp_pipeline
    from entity_recognition import entity_extractor
    from predict import prediction_engine
except ImportError as e:
    logger.warning(f"Some ML modules not available: {e}")

try:
    from database import database_manager
    from storage import storage_manager
    from security import security_manager
except ImportError as e:
    logger.warning(f"Some infrastructure modules not available: {e}")

class DocumentProcessor:
    """Main document processor orchestrator"""
    
    def __init__(self):
        self.processing_steps = [
            'initialization',
            'security_check',
            'file_validation',
            'ocr_processing',
            'text_analysis',
            'entity_extraction',
            'document_classification',
            'sentiment_analysis',
            'storage_operation',
            'database_storage',
            'cleanup'
        ]
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def process_document(self, file_path: str, document_id: str = None, 
                        options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single document through the complete pipeline"""
        # Generate document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())
        
        # Initialize processing context
        context = {
            'document_id': document_id,
            'file_path': file_path,
            'start_time': datetime.now().isoformat(),
            'options': options or {},
            'results': {},
            'processing_log': [],
            'success': True,
            'errors': []
        }
        
        logger.info(f"Starting document processing: {document_id}")
        
        # Execute processing pipeline
        for step in self.processing_steps:
            try:
                step_start = datetime.now()
                step_result = self._execute_step(step, context)
                step_duration = (datetime.now() - step_start).total_seconds()
                
                context['processing_log'].append({
                    'step': step,
                    'success': step_result['success'],
                    'duration': step_duration,
                    'timestamp': datetime.now().isoformat()
                })
                
                if not step_result['success']:
                    context['success'] = False
                    context['errors'].append({
                        'step': step,
                        'error': step_result.get('error', 'Unknown error')
                    })
                    
                    # Decide whether to continue or stop
                    if step in ['initialization', 'file_validation', 'security_check']:
                        # Critical steps - stop processing
                        break
                    else:
                        # Non-critical steps - log error and continue
                        logger.warning(f"Non-critical step failed: {step}")
                
                # Store step results
                context['results'][step] = step_result
                
            except Exception as e:
                logger.error(f"Step {step} failed with exception: {str(e)}")
                context['success'] = False
                context['errors'].append({
                    'step': step,
                    'error': str(e)
                })
                break
        
        # Finalize processing context
        context['end_time'] = datetime.now().isoformat()
        context['total_duration'] = (
            datetime.fromisoformat(context['end_time']) - 
            datetime.fromisoformat(context['start_time'])
        ).total_seconds()
        
        logger.info(f"Completed document processing: {document_id} (Success: {context['success']})")
        
        return context
    
    def _execute_step(self, step: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual processing step"""
        try:
            if step == 'initialization':
                return self._initialize_processing(context)
            elif step == 'security_check':
                return self._security_check(context)
            elif step == 'file_validation':
                return self._validate_file(context)
            elif step == 'ocr_processing':
                return self._extract_text(context)
            elif step == 'text_analysis':
                return self._analyze_text(context)
            elif step == 'entity_extraction':
                return self._extract_entities(context)
            elif step == 'document_classification':
                return self._classify_document(context)
            elif step == 'sentiment_analysis':
                return self._analyze_sentiment(context)
            elif step == 'storage_operation':
                return self._store_document(context)
            elif step == 'database_storage':
                return self._store_in_database(context)
            elif step == 'cleanup':
                return self._cleanup(context)
            else:
                return {'success': False, 'error': f'Unknown step: {step}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _initialize_processing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize processing"""
        try:
            # Create necessary directories
            config.create_directories()
            
            # Validate document ID
            if not context['document_id']:
                return {'success': False, 'error': 'No document ID provided'}
            
            return {
                'success': True,
                'message': 'Processing initialized',
                'document_id': context['document_id']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _security_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security checks"""
        try:
            file_path = context['file_path']
            
            # Check file exists
            if not os.path.exists(file_path):
                return {'success': False, 'error': 'File does not exist'}
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > config.app.max_file_size:
                return {'success': False, 'error': 'File too large'}
            
            # Basic security scan
            if security_manager:
                # Perform additional security checks if security manager is available
                pass
            
            return {
                'success': True,
                'file_size': file_size,
                'security_status': 'passed'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_file(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate file format and content"""
        try:
            file_path = context['file_path']
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Supported file types
            supported_types = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.txt', '.docx']
            
            if file_extension not in supported_types:
                return {
                    'success': False, 
                    'error': f'Unsupported file type: {file_extension}'
                }
            
            return {
                'success': True,
                'file_type': file_extension,
                'file_size': file_size,
                'supported': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_text(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text using OCR"""
        try:
            file_path = context['file_path']
            
            if not ocr_engine:
                return {'success': False, 'error': 'OCR engine not available'}
            
            # Determine OCR method based on file type
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.txt':
                # Read text file directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                
                return {
                    'success': True,
                    'text': extracted_text,
                    'method': 'direct_read',
                    'confidence': 1.0
                }
            
            elif file_extension == '.pdf':
                # Use PDF OCR
                ocr_results = ocr_engine.tesseract.extract_text_from_pdf(file_path)
                
                # Combine results from all pages
                all_text = []
                avg_confidence = 0
                
                for page_result in ocr_results:
                    if page_result.get('text'):
                        all_text.append(page_result['text'])
                        avg_confidence += page_result.get('confidence', 0)
                
                avg_confidence = avg_confidence / len(ocr_results) if ocr_results else 0
                
                return {
                    'success': True,
                    'text': '\n'.join(all_text),
                    'method': 'pdf_ocr',
                    'confidence': avg_confidence,
                    'pages': len(ocr_results)
                }
            
            else:
                # Use image OCR with fallback strategy
                ocr_result = ocr_engine.extract_text_with_fallback(file_path)
                
                return {
                    'success': True,
                    'text': ocr_result.get('text', ''),
                    'method': ocr_result.get('method', 'unknown'),
                    'confidence': ocr_result.get('confidence', 0)
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _analyze_text(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text using NLP"""
        try:
            ocr_result = context['results'].get('ocr_processing', {})
            text = ocr_result.get('text', '')
            
            if not text or len(text.strip()) < 10:
                return {'success': False, 'error': 'Insufficient text for analysis'}
            
            if not nlp_pipeline:
                return {'success': False, 'error': 'NLP pipeline not available'}
            
            # Process text with NLP pipeline
            nlp_results = nlp_pipeline.process_document(text, include_summary=True)
            
            return {
                'success': True,
                'nlp_results': nlp_results,
                'text_length': len(text),
                'processed_tokens': nlp_results.get('processed_tokens', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_entities(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from text"""
        try:
            ocr_result = context['results'].get('ocr_processing', {})
            text = ocr_result.get('text', '')
            
            if not text:
                return {'success': False, 'error': 'No text available for entity extraction'}
            
            if not entity_extractor:
                return {'success': False, 'error': 'Entity extractor not available'}
            
            # Determine document type from previous results
            nlp_results = context['results'].get('text_analysis', {}).get('nlp_results', {})
            document_type = nlp_results.get('document_type', {}).get('document_type', 'general')
            
            # Extract entities
            entity_results = entity_extractor.extract_all_entities(text, document_type)
            
            return {
                'success': True,
                'entities': entity_results,
                'document_type': document_type
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _classify_document(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document type"""
        try:
            ocr_result = context['results'].get('ocr_processing', {})
            text = ocr_result.get('text', '')
            
            if not text:
                return {'success': False, 'error': 'No text available for classification'}
            
            if not prediction_engine:
                return {'success': False, 'error': 'Prediction engine not available'}
            
            # Classify document type
            classification_result = prediction_engine.predict_document_type(text)
            
            return {
                'success': True,
                'classification': classification_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _analyze_sentiment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment"""
        try:
            ocr_result = context['results'].get('ocr_processing', {})
            text = ocr_result.get('text', '')
            
            if not text:
                return {'success': False, 'error': 'No text available for sentiment analysis'}
            
            if not prediction_engine:
                return {'success': False, 'error': 'Prediction engine not available'}
            
            # Analyze sentiment
            sentiment_result = prediction_engine.predict_sentiment(text)
            
            return {
                'success': True,
                'sentiment': sentiment_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _store_document(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store document in cloud storage"""
        try:
            file_path = context['file_path']
            document_id = context['document_id']
            
            if not storage_manager:
                return {'success': False, 'error': 'Storage manager not available'}
            
            # Prepare metadata
            metadata = {
                'document_type': context['results'].get('document_classification', {}).get('classification', {}).get('document_type', 'unknown'),
                'processing_timestamp': context['start_time'],
                'file_size': context['results'].get('file_validation', {}).get('file_size', 0),
                'confidence_scores': {}
            }
            
            # Store document
            storage_result = storage_manager.store_uploaded_document(file_path, document_id, metadata)
            
            return {
                'success': storage_result['success'],
                'storage_info': storage_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _store_in_database(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store processing results in database"""
        try:
            document_id = context['document_id']
            
            if not database_manager:
                return {'success': False, 'error': 'Database manager not available'}
            
            # Prepare document data
            document_data = {
                'filename': os.path.basename(context['file_path']),
                'file_type': context['results'].get('file_validation', {}).get('file_type', 'unknown'),
                'file_size': context['results'].get('file_validation', {}).get('file_size', 0),
                'document_type': context['results'].get('document_classification', {}).get('classification', {}).get('document_type', 'unknown'),
                'confidence_score': context['results'].get('document_classification', {}).get('classification', {}).get('confidence', 0),
                'extracted_text': context['results'].get('ocr_processing', {}).get('text', ''),
                'metadata': json.dumps(context['results'], default=str)
            }
            
            # Store in database
            storage_result = database_manager.store_processed_document(document_data)
            
            return {
                'success': True,
                'database_info': storage_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _cleanup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanup temporary files and resources"""
        try:
            # Clean up any temporary files
            temp_files = context.get('temp_files', [])
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
            
            return {
                'success': True,
                'message': 'Cleanup completed'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def process_document_async(self, file_path: str, document_id: str = None, 
                                   options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process document asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run processing in thread pool
        result = await loop.run_in_executor(
            self.executor,
            self.process_document,
            file_path,
            document_id,
            options
        )
        
        return result
    
    def batch_process_documents(self, file_paths: List[str], 
                              options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process multiple documents"""
        results = []
        
        for i, file_path in enumerate(file_paths):
            document_id = str(uuid.uuid4())
            
            logger.info(f"Processing document {i+1}/{len(file_paths)}: {file_path}")
            
            try:
                result = self.process_document(file_path, document_id, options)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed for {file_path}: {str(e)}")
                results.append({
                    'document_id': document_id,
                    'file_path': file_path,
                    'batch_index': i,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get overall processing system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'active',
            'available_modules': {
                'ocr_engine': bool(ocr_engine),
                'nlp_pipeline': bool(nlp_pipeline),
                'entity_extractor': bool(entity_extractor),
                'prediction_engine': bool(prediction_engine),
                'database_manager': bool(database_manager),
                'storage_manager': bool(storage_manager),
                'security_manager': bool(security_manager)
            },
            'processing_steps': self.processing_steps,
            'max_workers': self.executor._max_workers
        }

class DocumentProcessorAPI:
    """API interface for document processing"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
    
    def process_uploaded_file(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process uploaded file through complete pipeline"""
        try:
            result = self.processor.process_document(file_path, options=options)
            
            # Add API-specific metadata
            result['api_version'] = '1.0.0'
            result['processing_mode'] = 'complete_pipeline'
            
            return result
            
        except Exception as e:
            logger.error(f"API processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_processing_summary(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing summary"""
        if not processing_result.get('success'):
            return {
                'success': False,
                'error': 'Processing failed',
                'details': processing_result.get('errors', [])
            }
        
        # Extract key information
        summary = {
            'document_id': processing_result['document_id'],
            'success': processing_result['success'],
            'processing_time': processing_result.get('total_duration', 0),
            'steps_completed': len([step for step in processing_result.get('processing_log', []) if step['success']]),
            'total_steps': len(processing_result.get('processing_log', [])),
            'extracted_text_length': len(processing_result.get('results', {}).get('ocr_processing', {}).get('text', '')),
            'document_type': processing_result.get('results', {}).get('document_classification', {}).get('classification', {}).get('document_type', 'unknown'),
            'confidence': processing_result.get('results', {}).get('document_classification', {}).get('classification', {}).get('confidence', 0),
            'entities_found': 0,
            'sentiment': processing_result.get('results', {}).get('sentiment_analysis', {}).get('sentiment', {}).get('label', 'unknown')
        }
        
        # Count entities
        entities_result = processing_result.get('results', {}).get('entity_extraction', {}).get('entities', {})
        if isinstance(entities_result, dict):
            for entity_type, entities in entities_result.get('entities', {}).items():
                if isinstance(entities, list):
                    summary['entities_found'] += len(entities)
        
        return summary

# Initialize document processor
document_processor = DocumentProcessorAPI()