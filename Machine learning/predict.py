"""
Prediction module for making inferences with trained models
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging
import json
from datetime import datetime
import asyncio
import concurrent.futures
from config import config
from model import model_manager
from OCR import ocr_engine
from NLP import nlp_pipeline
from entity_recognition import entity_extractor

logger = logging.getLogger(__name__)

class PredictionEngine:
    """Main prediction engine that coordinates all model predictions"""
    
    def __init__(self):
        self.model_manager = model_manager
        self.ocr_engine = ocr_engine
        self.nlp_pipeline = nlp_pipeline
        self.entity_extractor = entity_extractor
        self.confidence_threshold = config.model.confidence_threshold
    
    def predict_document_type(self, text: str, method: str = 'ensemble') -> Dict[str, Any]:
        """Predict document type using specified method"""
        try:
            if method == 'ensemble':
                predictions = self.model_manager.predict_with_model('ensemble', [text])
            elif method in ['tensorflow', 'pytorch']:
                predictions = self.model_manager.predict_with_model(method, [text])
            else:
                # Fallback to NLP-based classification
                predictions = [self.nlp_pipeline.classifier.classify_document_type(text)]
            
            if predictions and len(predictions) > 0:
                prediction = predictions[0]
                prediction['method'] = method
                prediction['timestamp'] = datetime.now().isoformat()
                return prediction
            else:
                return {'error': 'No predictions generated', 'method': method}
                
        except Exception as e:
            logger.error(f"Document type prediction failed: {str(e)}")
            return {'error': str(e), 'method': method}
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment of the text"""
        try:
            result = self.nlp_pipeline.classifier.classify_sentiment(text)
            result['timestamp'] = datetime.now().isoformat()
            return result
        except Exception as e:
            logger.error(f"Sentiment prediction failed: {str(e)}")
            return {'error': str(e)}
    
    def extract_and_predict(self, text: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """Extract entities and make predictions on text"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'processing_steps': []
        }
        
        # Step 1: Document type prediction if not provided
        if not document_type:
            doc_type_result = self.predict_document_type(text)
            results['document_type_prediction'] = doc_type_result
            document_type = doc_type_result.get('document_type', 'general')
            results['processing_steps'].append('document_type_prediction')
        else:
            results['document_type'] = document_type
        
        # Step 2: Entity extraction
        try:
            entity_results = self.entity_extractor.extract_all_entities(text, document_type)
            results['entities'] = entity_results
            results['processing_steps'].append('entity_extraction')
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            results['entities'] = {'error': str(e)}
        
        # Step 3: NLP analysis
        try:
            nlp_results = self.nlp_pipeline.process_document(text, include_summary=True)
            results['nlp_analysis'] = nlp_results
            results['processing_steps'].append('nlp_analysis')
        except Exception as e:
            logger.error(f"NLP analysis failed: {str(e)}")
            results['nlp_analysis'] = {'error': str(e)}
        
        # Step 4: Sentiment prediction
        try:
            sentiment_result = self.predict_sentiment(text)
            results['sentiment'] = sentiment_result
            results['processing_steps'].append('sentiment_analysis')
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            results['sentiment'] = {'error': str(e)}
        
        return results
    
    async def process_document_async(self, text: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """Asynchronously process document"""
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive tasks in thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor, 
                self.extract_and_predict, 
                text, 
                document_type
            )
            result = await future
        
        return result
    
    def batch_predict(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process multiple texts in batches"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_results = []
            for j, text in enumerate(batch):
                try:
                    result = self.extract_and_predict(text)
                    result['batch_id'] = i // batch_size + 1
                    result['item_id'] = j + 1
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch item {j} processing failed: {str(e)}")
                    batch_results.append({
                        'error': str(e),
                        'batch_id': i // batch_size + 1,
                        'item_id': j + 1,
                        'timestamp': datetime.now().isoformat()
                    })
            
            results.extend(batch_results)
        
        return results
    
    def get_prediction_confidence(self, prediction_result: Dict[str, Any]) -> float:
        """Calculate overall confidence score for a prediction"""
        confidences = []
        
        # Document type confidence
        if 'document_type_prediction' in prediction_result:
            doc_conf = prediction_result['document_type_prediction'].get('confidence', 0)
            confidences.append(doc_conf)
        
        # Entity extraction confidence
        if 'entities' in prediction_result:
            entity_confidences = []
            entities = prediction_result['entities'].get('entities', {})
            
            for method_results in entities.values():
                if isinstance(method_results, list):
                    for entity in method_results:
                        if isinstance(entity, dict) and 'confidence' in entity:
                            entity_confidences.append(entity['confidence'])
            
            if entity_confidences:
                confidences.append(np.mean(entity_confidences))
        
        # Sentiment confidence
        if 'sentiment' in prediction_result:
            sent_conf = prediction_result['sentiment'].get('score', 0)
            confidences.append(sent_conf)
        
        # Calculate weighted average
        return float(np.mean(confidences)) if confidences else 0.0
    
    def filter_high_confidence_predictions(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results to only include high-confidence predictions"""
        filtered = []
        
        for result in results:
            confidence = self.get_prediction_confidence(result)
            if confidence >= self.confidence_threshold:
                result['overall_confidence'] = confidence
                result['high_confidence'] = True
                filtered.append(result)
            else:
                result['overall_confidence'] = confidence
                result['high_confidence'] = False
                result['filtered_reason'] = 'Low confidence'
        
        return filtered

class ImageToTextPredictor:
    """Predictor for processing images and extracting text predictions"""
    
    def __init__(self):
        self.prediction_engine = PredictionEngine()
        self.ocr_engine = ocr_engine
    
    def process_image(self, image_path: str, ocr_method: str = 'hybrid') -> Dict[str, Any]:
        """Process image and make predictions on extracted text"""
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'ocr_method': ocr_method
        }
        
        # Step 1: OCR text extraction
        try:
            if ocr_method == 'hybrid':
                ocr_result = self.ocr_engine.extract_text_with_fallback(image_path)
            elif ocr_method == 'tesseract':
                ocr_result = self.ocr_engine.tesseract.extract_text(image_path)
            elif ocr_method == 'google_vision':
                ocr_result = self.ocr_engine.google_vision.extract_text(image_path)
            else:
                ocr_result = self.ocr_engine.extract_text(image_path, use_best=True)
            
            results['ocr_result'] = ocr_result
            extracted_text = ocr_result.get('text', '')
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            results['ocr_result'] = {'error': str(e)}
            extracted_text = ''
        
        # Step 2: Process extracted text if available
        if extracted_text and len(extracted_text.strip()) > 10:
            try:
                prediction_results = self.prediction_engine.extract_and_predict(extracted_text)
                results['predictions'] = prediction_results
                results['success'] = True
            except Exception as e:
                logger.error(f"Prediction processing failed: {str(e)}")
                results['predictions'] = {'error': str(e)}
                results['success'] = False
        else:
            results['predictions'] = {'warning': 'Insufficient text extracted for predictions'}
            results['success'] = False
        
        return results
    
    def batch_process_images(self, image_paths: List[str], ocr_method: str = 'hybrid') -> List[Dict[str, Any]]:
        """Process multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.process_image(image_path, ocr_method)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Image processing failed for {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'image_index': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results

class PredictionAPI:
    """API interface for making predictions"""
    
    def __init__(self):
        self.text_predictor = PredictionEngine()
        self.image_predictor = ImageToTextPredictor()
    
    def predict_text(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """API endpoint for text prediction"""
        options = options or {}
        
        try:
            document_type = options.get('document_type')
            include_entities = options.get('include_entities', True)
            include_sentiment = options.get('include_sentiment', True)
            
            if include_entities:
                result = self.text_predictor.extract_and_predict(text, document_type)
            else:
                # Simplified prediction without entities
                result = {
                    'document_type_prediction': self.text_predictor.predict_document_type(text),
                    'timestamp': datetime.now().isoformat()
                }
                
                if include_sentiment:
                    result['sentiment'] = self.text_predictor.predict_sentiment(text)
            
            # Add confidence score
            result['overall_confidence'] = self.text_predictor.get_prediction_confidence(result)
            
            return {
                'success': True,
                'data': result,
                'options_used': options
            }
            
        except Exception as e:
            logger.error(f"Text prediction API failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'options_used': options
            }
    
    def predict_image(self, image_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """API endpoint for image prediction"""
        options = options or {}
        
        try:
            ocr_method = options.get('ocr_method', 'hybrid')
            result = self.image_predictor.process_image(image_path, ocr_method)
            
            return {
                'success': result.get('success', False),
                'data': result,
                'options_used': options
            }
            
        except Exception as e:
            logger.error(f"Image prediction API failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'options_used': options
            }
    
    def batch_predict_texts(self, texts: List[str], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """API endpoint for batch text prediction"""
        options = options or {}
        
        try:
            batch_size = options.get('batch_size', 10)
            results = self.text_predictor.batch_predict(texts, batch_size)
            
            # Filter high confidence if requested
            if options.get('filter_high_confidence', False):
                results = self.text_predictor.filter_high_confidence_predictions(results)
            
            return {
                'success': True,
                'data': {
                    'results': results,
                    'total_processed': len(texts),
                    'batch_size': batch_size
                },
                'options_used': options
            }
            
        except Exception as e:
            logger.error(f"Batch text prediction API failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'options_used': options
            }

# Initialize prediction instances
prediction_engine = PredictionEngine()
image_to_text_predictor = ImageToTextPredictor()
prediction_api = PredictionAPI()