"""
Optical Character Recognition module with Tesseract and Google Cloud Vision AI integration
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from google.cloud import vision
from google.oauth2 import service_account
import base64
import io
import logging
from typing import List, Dict, Optional, Union
from config import config

logger = logging.getLogger(__name__)

class TesseractOCR:
    """Tesseract OCR implementation"""
    
    def __init__(self):
        self.tesseract_cmd = config.ocr.tesseract_cmd
        self.tesseract_config = config.ocr.tesseract_config
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        return closing
    
    def extract_text(self, image_path: str, language: str = 'eng') -> Dict[str, Union[str, float]]:
        """Extract text from image using Tesseract"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text with confidence
            data = pytesseract.image_to_data(
                processed_image, 
                lang=language, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter out low confidence detections
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image, 
                lang=language, 
                config=self.tesseract_config
            )
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'method': 'tesseract',
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            return {'text': '', 'confidence': 0.0, 'error': str(e)}
    
    def extract_text_from_pdf(self, pdf_path: str, language: str = 'eng') -> List[Dict[str, Union[str, float]]]:
        """Extract text from PDF using Tesseract"""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            results = []
            
            for i, image in enumerate(images):
                # Save temporary image
                temp_path = f"temp/page_{i}.png"
                image.save(temp_path)
                
                # Extract text
                result = self.extract_text(temp_path, language)
                result['page'] = i + 1
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"PDF OCR failed: {str(e)}")
            return [{'text': '', 'confidence': 0.0, 'error': str(e)}]

class GoogleVisionOCR:
    """Google Cloud Vision API OCR implementation"""
    
    def __init__(self):
        self.project_id = config.cloud.google_project_id
        self.credentials_path = config.cloud.google_credentials_path
        
        # Initialize client
        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            self.client = vision.ImageAnnotatorClient()
    
    def extract_text(self, image_path: str) -> Dict[str, Union[str, float]]:
        """Extract text using Google Vision API"""
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Detect text
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                full_text = texts[0].description
                confidence = self._calculate_confidence(texts)
                
                return {
                    'text': full_text,
                    'confidence': confidence,
                    'method': 'google_vision',
                    'annotations': len(texts)
                }
            else:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'method': 'google_vision',
                    'annotations': 0
                }
                
        except Exception as e:
            logger.error(f"Google Vision OCR failed: {str(e)}")
            return {'text': '', 'confidence': 0.0, 'error': str(e)}
    
    def extract_document_text(self, image_path: str) -> Dict[str, Union[str, List]]:
        """Extract structured document text using Google Vision API"""
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Detect document text
            response = self.client.document_text_detection(image=image)
            document = response.full_text_annotation
            
            if document:
                pages = []
                for page in document.pages:
                    page_data = {
                        'width': page.width,
                        'height': page.height,
                        'blocks': []
                    }
                    
                    for block in page.blocks:
                        block_text = ""
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                word_text = ''.join([symbol.text for symbol in word.symbols])
                                block_text += word_text + " "
                        
                        page_data['blocks'].append({
                            'text': block_text.strip(),
                            'confidence': block.confidence
                        })
                    
                    pages.append(page_data)
                
                return {
                    'text': document.text,
                    'pages': pages,
                    'method': 'google_vision_document'
                }
            else:
                return {
                    'text': '',
                    'pages': [],
                    'method': 'google_vision_document'
                }
                
        except Exception as e:
            logger.error(f"Google Vision Document OCR failed: {str(e)}")
            return {'text': '', 'pages': [], 'error': str(e)}
    
    def _calculate_confidence(self, annotations) -> float:
        """Calculate average confidence from annotations"""
        if len(annotations) <= 1:
            return 1.0
        
        confidences = []
        for annotation in annotations[1:]:  # Skip the first one (full text)
            if hasattr(annotation, 'confidence'):
                confidences.append(annotation.confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.0

class HybridOCR:
    """Hybrid OCR that combines multiple OCR engines"""
    
    def __init__(self):
        self.tesseract = TesseractOCR()
        self.google_vision = GoogleVisionOCR()
    
    def extract_text(self, image_path: str, use_best: bool = True) -> Dict[str, Union[str, float]]:
        """Extract text using multiple OCR engines and return best result"""
        results = []
        
        # Try Tesseract
        tesseract_result = self.tesseract.extract_text(image_path)
        results.append(tesseract_result)
        
        # Try Google Vision
        try:
            vision_result = self.google_vision.extract_text(image_path)
            results.append(vision_result)
        except Exception as e:
            logger.warning(f"Google Vision OCR unavailable: {str(e)}")
        
        if use_best:
            # Return result with highest confidence
            best_result = max(results, key=lambda x: x.get('confidence', 0))
            return best_result
        else:
            # Return all results
            return {'results': results}
    
    def extract_text_with_fallback(self, image_path: str) -> Dict[str, Union[str, float]]:
        """Extract text with fallback strategy"""
        # Try Google Vision first (usually more accurate)
        try:
            result = self.google_vision.extract_text(image_path)
            if result.get('confidence', 0) > 0.5:
                return result
        except Exception as e:
            logger.warning(f"Google Vision failed, falling back to Tesseract: {str(e)}")
        
        # Fallback to Tesseract
        return self.tesseract.extract_text(image_path)

# Initialize OCR instances
ocr_engine = HybridOCR()