"""
Natural Language Processing module for document analysis and text processing
"""
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from config import config

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_pipeline(self, text: str) -> List[str]:
        """Complete preprocessing pipeline"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        filtered_tokens = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize(filtered_tokens)
        return lemmatized_tokens

class NamedEntityRecognition:
    """Named Entity Recognition using spaCy and NLTK"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_entities_spacy(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return {'entities': [], 'error': 'spaCy model not available'}
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'confidence', 1.0)
            })
        
        return {
            'entities': entities,
            'method': 'spacy',
            'model': 'en_core_web_sm'
        }
    
    def extract_entities_nltk(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract named entities using NLTK"""
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        
        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity_text = ' '.join([token for token, pos in chunk.leaves()])
                entities.append({
                    'text': entity_text,
                    'label': chunk.label(),
                    'method': 'nltk'
                })
        
        return {
            'entities': entities,
            'method': 'nltk'
        }

class TextSummarization:
    """Text summarization using various techniques"""
    
    def __init__(self):
        self.summarizer = None
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            logger.warning(f"Summarization model not available: {str(e)}")
    
    def extractive_summarization(self, text: str, num_sentences: int = 3) -> str:
        """Extractive summarization using TF-IDF"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = np.mean(tfidf_matrix.toarray(), axis=1)
        
        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices.sort()
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def abstractive_summarization(self, text: str, max_length: int = 150) -> str:
        """Abstractive summarization using transformer models"""
        if not self.summarizer:
            return self.extractive_summarization(text)
        
        try:
            # Split text if too long
            max_input_length = 1024
            if len(text) > max_input_length:
                # Split into chunks
                chunks = [text[i:i+max_input_length] for i in range(0, len(text), max_input_length)]
                summaries = []
                
                for chunk in chunks:
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                    
                    summary = self.summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                
                return ' '.join(summaries)
            else:
                summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
                return summary[0]['summary_text']
                
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {str(e)}")
            return self.extractive_summarization(text)

class TextClassification:
    """Text classification and categorization"""
    
    def __init__(self):
        self.classifier = None
        try:
            self.classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            logger.warning(f"Classification model not available: {str(e)}")
    
    def classify_sentiment(self, text: str) -> Dict[str, Any]:
        """Classify text sentiment"""
        if not self.classifier:
            return {'label': 'UNKNOWN', 'score': 0.0, 'error': 'Model not available'}
        
        try:
            result = self.classifier(text)
            return {
                'label': result[0]['label'],
                'score': result[0]['score'],
                'method': 'transformer'
            }
        except Exception as e:
            logger.error(f"Sentiment classification failed: {str(e)}")
            return {'label': 'ERROR', 'score': 0.0, 'error': str(e)}
    
    def classify_document_type(self, text: str) -> Dict[str, Any]:
        """Classify document type based on content patterns"""
        text_lower = text.lower()
        
        # Define patterns for different document types
        patterns = {
            'invoice': ['invoice', 'bill', 'amount due', 'payment', 'total'],
            'contract': ['agreement', 'terms', 'conditions', 'parties', 'signature'],
            'resume': ['experience', 'education', 'skills', 'employment', 'qualifications'],
            'report': ['summary', 'analysis', 'findings', 'conclusion', 'recommendation'],
            'email': ['from:', 'to:', 'subject:', 'dear', 'sincerely'],
            'letter': ['dear', 'sincerely', 'yours truly', 'best regards']
        }
        
        scores = {}
        for doc_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score / len(keywords)
        
        best_match = max(scores, key=scores.get)
        confidence = scores[best_match]
        
        return {
            'document_type': best_match,
            'confidence': confidence,
            'scores': scores,
            'method': 'pattern_matching'
        }

class TextSimilarity:
    """Text similarity and clustering"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            return 0.0
    
    def cluster_documents(self, texts: List[str], num_clusters: int = 3) -> Dict[str, Any]:
        """Cluster documents using K-means"""
        try:
            if len(texts) < num_clusters:
                num_clusters = len(texts)
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group documents by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'index': i,
                    'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i]
                })
            
            return {
                'clusters': clusters,
                'num_clusters': num_clusters,
                'method': 'kmeans'
            }
            
        except Exception as e:
            logger.error(f"Document clustering failed: {str(e)}")
            return {'clusters': {}, 'error': str(e)}

class NLPPipeline:
    """Complete NLP processing pipeline"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.ner = NamedEntityRecognition()
        self.summarizer = TextSummarization()
        self.classifier = TextClassification()
        self.similarity = TextSimilarity()
    
    def process_document(self, text: str, include_summary: bool = True) -> Dict[str, Any]:
        """Complete NLP processing of a document"""
        results = {
            'original_length': len(text),
            'processed_tokens': len(self.preprocessor.preprocess_pipeline(text)),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Named Entity Recognition
        try:
            results['entities'] = self.ner.extract_entities_spacy(text)
        except Exception as e:
            logger.error(f"NER processing failed: {str(e)}")
            results['entities'] = {'error': str(e)}
        
        # Document Classification
        try:
            results['document_type'] = self.classifier.classify_document_type(text)
            results['sentiment'] = self.classifier.classify_sentiment(text)
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            results['classification'] = {'error': str(e)}
        
        # Text Summarization
        if include_summary:
            try:
                results['summary'] = {
                    'extractive': self.summarizer.extractive_summarization(text),
                    'abstractive': self.summarizer.abstractive_summarization(text)
                }
            except Exception as e:
                logger.error(f"Summarization failed: {str(e)}")
                results['summary'] = {'error': str(e)}
        
        return results
    
    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents"""
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing document {i+1}/{len(texts)}")
            result = self.process_document(text)
            result['document_id'] = i
            results.append(result)
        
        # Add clustering analysis
        try:
            cluster_results = self.similarity.cluster_documents(texts)
            return {
                'individual_results': results,
                'cluster_analysis': cluster_results
            }
        except Exception as e:
            logger.error(f"Batch clustering failed: {str(e)}")
            return results

# Initialize NLP pipeline
nlp_pipeline = NLPPipeline()