"""
Advanced Entity Recognition module for extracting structured information from documents
"""
import re
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, date
from config import config

logger = logging.getLogger(__name__)

class PatternBasedExtractor:
    """Pattern-based entity extraction using regular expressions"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'currency': r'\$[\d,]+\.?\d*',
            'percentage': r'\d+\.?\d*%',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'invoice_number': r'(?i)(?:invoice|inv)[#\s]*:?\s*([A-Z0-9-]+)',
            'po_number': r'(?i)(?:purchase order|po)[#\s]*:?\s*([A-Z0-9-]+)',
            'tax_id': r'(?i)(?:tax id|ein)[#\s]*:?\s*(\d{2}-\d{7})'
        }
    
    def extract_patterns(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all pattern-based entities from text"""
        results = {}
        
        for pattern_name, pattern in self.patterns.items():
            matches = []
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'pattern': pattern_name,
                    'confidence': 0.9  # High confidence for regex matches
                })
            
            if matches:
                results[pattern_name] = matches
        
        return results
    
    def extract_addresses(self, text: str) -> List[Dict[str, Any]]:
        """Extract addresses using complex patterns"""
        # Address pattern - simplified but covers most cases
        address_pattern = r'\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)(?:\s+[A-Za-z0-9\s,.-]+)*(?:\s+\d{5}(?:-\d{4})?)?'
        
        addresses = []
        for match in re.finditer(address_pattern, text, re.IGNORECASE):
            addresses.append({
                'text': match.group().strip(),
                'start': match.start(),
                'end': match.end(),
                'type': 'address',
                'confidence': 0.8
            })
        
        return addresses
    
    def extract_financial_info(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract financial information"""
        financial_patterns = {
            'account_number': r'(?i)(?:account|acct)[#\s]*:?\s*([0-9-]+)',
            'routing_number': r'(?i)(?:routing|aba)[#\s]*:?\s*([0-9]{9})',
            'total_amount': r'(?i)(?:total|amount due)[:\s]*\$?([\d,]+\.?\d*)',
            'subtotal': r'(?i)subtotal[:\s]*\$?([\d,]+\.?\d*)',
            'tax_amount': r'(?i)(?:tax|vat)[:\s]*\$?([\d,]+\.?\d*)',
            'discount': r'(?i)discount[:\s]*\$?([\d,]+\.?\d*)'
        }
        
        results = {}
        for pattern_name, pattern in financial_patterns.items():
            matches = []
            for match in re.finditer(pattern, text):
                matches.append({
                    'text': match.group(),
                    'value': match.group(1) if match.groups() else match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': pattern_name,
                    'confidence': 0.85
                })
            
            if matches:
                results[pattern_name] = matches
        
        return results

class TransformerEntityExtractor:
    """Entity extraction using transformer models"""
    
    def __init__(self):
        self.ner_pipeline = None
        try:
            self.ner_pipeline = pipeline(
                "ner", 
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
        except Exception as e:
            logger.warning(f"Transformer NER model not available: {str(e)}")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using transformer model"""
        if not self.ner_pipeline:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            
            # Process and normalize results
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    'text': entity['word'],
                    'label': entity['entity_group'],
                    'confidence': entity['score'],
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0),
                    'method': 'transformer'
                })
            
            return processed_entities
            
        except Exception as e:
            logger.error(f"Transformer entity extraction failed: {str(e)}")
            return []

class DocumentSpecificExtractor:
    """Document-specific entity extractors for different document types"""
    
    def extract_invoice_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities specific to invoices"""
        entities = {
            'invoice_info': {},
            'vendor_info': {},
            'customer_info': {},
            'line_items': [],
            'financial_info': {}
        }
        
        # Invoice number
        invoice_match = re.search(r'(?i)invoice[#\s]*:?\s*([A-Z0-9-]+)', text)
        if invoice_match:
            entities['invoice_info']['number'] = invoice_match.group(1)
        
        # Date patterns
        date_match = re.search(r'(?i)(?:date|invoice date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})', text)
        if date_match:
            entities['invoice_info']['date'] = date_match.group(1)
        
        # Due date
        due_date_match = re.search(r'(?i)(?:due date|payment due)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})', text)
        if due_date_match:
            entities['invoice_info']['due_date'] = due_date_match.group(1)
        
        # Total amount
        total_match = re.search(r'(?i)(?:total|amount due|grand total)[:\s]*\$?([\d,]+\.?\d*)', text)
        if total_match:
            entities['financial_info']['total'] = total_match.group(1)
        
        return entities
    
    def extract_resume_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities specific to resumes"""
        entities = {
            'personal_info': {},
            'education': [],
            'experience': [],
            'skills': [],
            'certifications': []
        }
        
        # Name (usually at the beginning)
        lines = text.split('\n')
        if lines:
            # First non-empty line is likely the name
            for line in lines:
                if line.strip():
                    entities['personal_info']['name'] = line.strip()
                    break
        
        # Education section
        education_match = re.search(r'(?i)(education.*?)(?=experience|skills|$)', text, re.DOTALL)
        if education_match:
            education_text = education_match.group(1)
            # Extract degrees
            degree_matches = re.findall(r'(?i)(bachelor|master|phd|doctorate|associate|certificate).*?(?=\n|\d{4})', education_text)
            entities['education'] = degree_matches
        
        # Experience section
        exp_match = re.search(r'(?i)(experience.*?)(?=education|skills|$)', text, re.DOTALL)
        if exp_match:
            exp_text = exp_match.group(1)
            # Extract job titles and companies
            job_matches = re.findall(r'(?i)([a-zA-Z\s]+(?:engineer|manager|analyst|developer|consultant|specialist|coordinator))', exp_text)
            entities['experience'] = job_matches
        
        return entities
    
    def extract_contract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities specific to contracts"""
        entities = {
            'parties': [],
            'dates': {},
            'terms': {},
            'clauses': []
        }
        
        # Contract parties
        party_patterns = [
            r'(?i)between\s+(.+?)\s+and\s+(.+?)(?:\s+hereby|\s+agree|\s+enter)',
            r'(?i)party\s+["\']?([^"\']+)["\']?\s+and\s+["\']?([^"\']+)["\']?'
        ]
        
        for pattern in party_patterns:
            match = re.search(pattern, text)
            if match:
                entities['parties'] = [match.group(1).strip(), match.group(2).strip()]
                break
        
        # Effective date
        effective_date = re.search(r'(?i)effective(?:\s+date)?[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})', text)
        if effective_date:
            entities['dates']['effective_date'] = effective_date.group(1)
        
        # Term duration
        term_match = re.search(r'(?i)term.*?(\d+)\s*(year|month|day)', text)
        if term_match:
            entities['terms']['duration'] = f"{term_match.group(1)} {term_match.group(2)}"
        
        return entities

class EntityValidator:
    """Validate and clean extracted entities"""
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_phone(self, phone: str) -> Tuple[bool, str]:
        """Validate and normalize phone number"""
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)
        
        if len(digits_only) == 10:
            return True, f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
        elif len(digits_only) == 11 and digits_only.startswith('1'):
            return True, f"1-({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:]}"
        else:
            return False, phone
    
    def validate_date(self, date_str: str) -> Tuple[bool, Optional[datetime]]:
        """Validate and parse date"""
        date_formats = [
            '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', 
            '%m-%d-%Y', '%d-%m-%Y', '%Y/%m/%d'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return True, parsed_date
            except ValueError:
                continue
        
        return False, None
    
    def clean_currency(self, currency_str: str) -> float:
        """Clean and convert currency string to float"""
        # Remove currency symbols and commas
        cleaned = re.sub(r'[\$,]', '', currency_str)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

class ComprehensiveEntityExtractor:
    """Comprehensive entity extraction combining multiple approaches"""
    
    def __init__(self):
        self.pattern_extractor = PatternBasedExtractor()
        self.transformer_extractor = TransformerEntityExtractor()
        self.document_extractor = DocumentSpecificExtractor()
        self.validator = EntityValidator()
    
    def extract_all_entities(self, text: str, document_type: str = 'general') -> Dict[str, Any]:
        """Extract entities using all available methods"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'document_type': document_type,
            'text_length': len(text),
            'entities': {}
        }
        
        # Pattern-based extraction
        try:
            pattern_entities = self.pattern_extractor.extract_patterns(text)
            results['entities']['patterns'] = pattern_entities
        except Exception as e:
            logger.error(f"Pattern extraction failed: {str(e)}")
            results['entities']['patterns'] = {'error': str(e)}
        
        # Transformer-based extraction
        try:
            transformer_entities = self.transformer_extractor.extract_entities(text)
            results['entities']['transformer'] = transformer_entities
        except Exception as e:
            logger.error(f"Transformer extraction failed: {str(e)}")
            results['entities']['transformer'] = {'error': str(e)}
        
        # Document-specific extraction
        try:
            if document_type == 'invoice':
                doc_entities = self.document_extractor.extract_invoice_entities(text)
            elif document_type == 'resume':
                doc_entities = self.document_extractor.extract_resume_entities(text)
            elif document_type == 'contract':
                doc_entities = self.document_extractor.extract_contract_entities(text)
            else:
                doc_entities = {}
            
            results['entities']['document_specific'] = doc_entities
        except Exception as e:
            logger.error(f"Document-specific extraction failed: {str(e)}")
            results['entities']['document_specific'] = {'error': str(e)}
        
        # Validation and cleaning
        results['entities']['validated'] = self._validate_entities(results['entities'])
        
        return results
    
    def _validate_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted entities"""
        validated = {}
        
        # Validate patterns
        if 'patterns' in entities and isinstance(entities['patterns'], dict):
            validated['emails'] = []
            validated['phones'] = []
            validated['dates'] = []
            validated['currencies'] = []
            
            # Validate emails
            if 'email' in entities['patterns']:
                for email_entity in entities['patterns']['email']:
                    if self.validator.validate_email(email_entity['text']):
                        validated['emails'].append(email_entity)
            
            # Validate phones
            if 'phone' in entities['patterns']:
                for phone_entity in entities['patterns']['phone']:
                    is_valid, normalized = self.validator.validate_phone(phone_entity['text'])
                    if is_valid:
                        phone_entity['normalized'] = normalized
                        validated['phones'].append(phone_entity)
            
            # Validate dates
            if 'date' in entities['patterns']:
                for date_entity in entities['patterns']['date']:
                    is_valid, parsed_date = self.validator.validate_date(date_entity['text'])
                    if is_valid:
                        date_entity['parsed'] = parsed_date.isoformat() if parsed_date else None
                        validated['dates'].append(date_entity)
            
            # Clean currencies
            if 'currency' in entities['patterns']:
                for currency_entity in entities['patterns']['currency']:
                    cleaned_amount = self.validator.clean_currency(currency_entity['text'])
                    currency_entity['amount'] = cleaned_amount
                    validated['currencies'].append(currency_entity)
        
        return validated
    
    def get_entity_summary(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of extracted entities"""
        summary = {
            'total_entities': 0,
            'entity_types': set(),
            'confidence_scores': [],
            'extraction_methods': set()
        }
        
        def count_entities(entity_dict, prefix=''):
            count = 0
            if isinstance(entity_dict, dict):
                for key, value in entity_dict.items():
                    if isinstance(value, list):
                        count += len(value)
                        for item in value:
                            if isinstance(item, dict):
                                if 'confidence' in item:
                                    summary['confidence_scores'].append(item['confidence'])
                                if 'method' in item:
                                    summary['extraction_methods'].add(item['method'])
                                if 'type' in item:
                                    summary['entity_types'].add(item['type'])
                                elif 'label' in item:
                                    summary['entity_types'].add(item['label'])
                        summary['entity_types'].add(key)
                    elif isinstance(value, dict):
                        count += count_entities(value, f"{prefix}.{key}" if prefix else key)
            return count
        
        if 'entities' in extraction_results:
            summary['total_entities'] = count_entities(extraction_results['entities'])
        
        summary['entity_types'] = list(summary['entity_types'])
        summary['extraction_methods'] = list(summary['extraction_methods'])
        
        if summary['confidence_scores']:
            summary['avg_confidence'] = np.mean(summary['confidence_scores'])
            summary['min_confidence'] = np.min(summary['confidence_scores'])
            summary['max_confidence'] = np.max(summary['confidence_scores'])
        
        return summary

# Initialize entity extractor
entity_extractor = ComprehensiveEntityExtractor()