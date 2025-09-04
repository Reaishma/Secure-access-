"""
Machine Learning Models module using TensorFlow, Keras, and PyTorch
"""
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pickle
import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from config import config

logger = logging.getLogger(__name__)

class DocumentDataset(Dataset):
    """PyTorch Dataset for document classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Simple text-to-vector encoding
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }

class TensorFlowDocumentClassifier:
    """TensorFlow/Keras model for document classification"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100, max_length: int = 500):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
    
    def build_model(self, num_classes: int) -> tf.keras.Model:
        """Build TensorFlow model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            tf.keras.layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_texts(self, texts: List[str]) -> np.ndarray:
        """Preprocess texts for training"""
        if not self.tokenizer:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=self.vocab_size,
                oov_token="<OOV>"
            )
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_length)
    
    def train(self, texts: List[str], labels: List[str], epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the model"""
        try:
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(labels)
            num_classes = len(self.label_encoder.classes_)
            
            # Preprocess texts
            X = self.preprocess_texts(texts)
            
            # Build model if not exists
            if not self.model:
                self.build_model(num_classes)
            
            # Train model
            history = self.model.fit(
                X, encoded_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            
            return {
                'success': True,
                'history': history.history,
                'classes': self.label_encoder.classes_.tolist(),
                'model_summary': str(self.model.summary())
            }
            
        except Exception as e:
            logger.error(f"TensorFlow model training failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions"""
        if not self.model:
            return [{'error': 'Model not trained'}]
        
        try:
            X = self.preprocess_texts(texts)
            predictions = self.model.predict(X)
            
            results = []
            for i, pred in enumerate(predictions):
                class_idx = np.argmax(pred)
                confidence = float(np.max(pred))
                class_name = self.label_encoder.inverse_transform([class_idx])[0]
                
                results.append({
                    'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                    'predicted_class': class_name,
                    'confidence': confidence,
                    'all_probabilities': {
                        self.label_encoder.inverse_transform([j])[0]: float(prob)
                        for j, prob in enumerate(pred)
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"TensorFlow prediction failed: {str(e)}")
            return [{'error': str(e)}]
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
            # Save tokenizer and label encoder
            with open(f"{filepath}_tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f)
            with open(f"{filepath}_label_encoder.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            # Load tokenizer and label encoder
            with open(f"{filepath}_tokenizer.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(f"{filepath}_label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

class PyTorchDocumentClassifier(nn.Module):
    """PyTorch model for document classification"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100, hidden_dim: int = 128, num_classes: int = 5):
        super(PyTorchDocumentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use the last hidden state
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out

class PyTorchTrainer:
    """Trainer class for PyTorch models"""
    
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.label_encoder = LabelEncoder()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10, lr: float = 0.001) -> Dict[str, Any]:
        """Train the PyTorch model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            
            for batch in train_loader:
                if isinstance(batch, dict) and 'input_ids' in batch:
                    inputs = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    continue
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict) and 'input_ids' in batch:
                        inputs = batch['input_ids'].to(self.device)
                        labels = batch['labels'].to(self.device)
                    else:
                        continue
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_loss = total_train_loss / len(train_loader)
            train_acc = correct_train / total_train
            val_loss = total_val_loss / len(val_loader)
            val_acc = correct_val / total_val
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def predict(self, data_loader: DataLoader) -> List[Dict[str, Any]]:
        """Make predictions with PyTorch model"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict) and 'input_ids' in batch:
                    inputs = batch['input_ids'].to(self.device)
                else:
                    continue
                
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                for i in range(len(predicted)):
                    predictions.append({
                        'predicted_class': predicted[i].item(),
                        'confidence': probabilities[i][predicted[i]].item(),
                        'all_probabilities': probabilities[i].cpu().numpy().tolist()
                    })
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save PyTorch model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_classes': self.model.num_classes
            },
            'label_encoder': self.label_encoder
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load PyTorch model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.label_encoder = checkpoint['label_encoder']
            return True
        except Exception as e:
            logger.error(f"PyTorch model loading failed: {str(e)}")
            return False

class EnsembleModel:
    """Ensemble model combining TensorFlow and PyTorch models"""
    
    def __init__(self):
        self.tf_model = TensorFlowDocumentClassifier()
        self.pytorch_model = None
        self.pytorch_trainer = None
        self.weights = {'tensorflow': 0.5, 'pytorch': 0.5}
    
    def train_ensemble(self, texts: List[str], labels: List[str], epochs: int = 10) -> Dict[str, Any]:
        """Train both models in the ensemble"""
        results = {}
        
        # Train TensorFlow model
        tf_results = self.tf_model.train(texts, labels, epochs=epochs)
        results['tensorflow'] = tf_results
        
        # Train PyTorch model
        try:
            # Prepare data for PyTorch
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            
            # Create simple word-to-index mapping
            vocab = set()
            for text in texts:
                vocab.update(text.split())
            vocab = list(vocab)
            word_to_idx = {word: idx for idx, word in enumerate(vocab)}
            
            # Convert texts to sequences
            sequences = []
            for text in texts:
                seq = [word_to_idx.get(word, 0) for word in text.split()[:100]]  # Limit to 100 words
                seq += [0] * (100 - len(seq))  # Pad to 100
                sequences.append(seq)
            
            # Create PyTorch model
            self.pytorch_model = PyTorchDocumentClassifier(
                vocab_size=len(vocab),
                num_classes=len(label_encoder.classes_)
            )
            self.pytorch_trainer = PyTorchTrainer(self.pytorch_model)
            self.pytorch_trainer.label_encoder = label_encoder
            
            # Create dataset and dataloaders
            dataset = DocumentDataset(sequences, encoded_labels.tolist())
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            pytorch_results = self.pytorch_trainer.train(train_loader, val_loader, epochs=epochs)
            results['pytorch'] = pytorch_results
            
        except Exception as e:
            logger.error(f"PyTorch training failed: {str(e)}")
            results['pytorch'] = {'error': str(e)}
        
        return results
    
    def predict_ensemble(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make ensemble predictions"""
        results = []
        
        # Get TensorFlow predictions
        tf_predictions = self.tf_model.predict(texts)
        
        # Get PyTorch predictions if available
        pytorch_predictions = []
        if self.pytorch_trainer and self.pytorch_model:
            try:
                # Convert texts to sequences (simplified)
                sequences = []
                for text in texts:
                    seq = [hash(word) % 1000 for word in text.split()[:100]]
                    seq += [0] * (100 - len(seq))
                    sequences.append(seq)
                
                dataset = DocumentDataset(sequences, [0] * len(sequences))  # Dummy labels
                data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
                pytorch_predictions = self.pytorch_trainer.predict(data_loader)
            except Exception as e:
                logger.error(f"PyTorch prediction failed: {str(e)}")
        
        # Combine predictions
        for i, text in enumerate(texts):
            ensemble_result = {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'tensorflow_prediction': tf_predictions[i] if i < len(tf_predictions) else None,
                'pytorch_prediction': pytorch_predictions[i] if i < len(pytorch_predictions) else None,
                'ensemble_method': 'weighted_average'
            }
            
            results.append(ensemble_result)
        
        return results

class ModelManager:
    """Manager for all ML models"""
    
    def __init__(self):
        self.models = {
            'tensorflow': TensorFlowDocumentClassifier(),
            'pytorch': None,
            'ensemble': EnsembleModel()
        }
        self.model_paths = {
            'tensorflow': config.model.tensorflow_model_path,
            'pytorch': config.model.pytorch_model_path
        }
    
    def get_model(self, model_type: str):
        """Get model by type"""
        return self.models.get(model_type)
    
    def train_model(self, model_type: str, texts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]:
        """Train specific model"""
        model = self.get_model(model_type)
        if not model:
            return {'error': f'Model type {model_type} not found'}
        
        try:
            if model_type == 'ensemble':
                return model.train_ensemble(texts, labels, **kwargs)
            else:
                return model.train(texts, labels, **kwargs)
        except Exception as e:
            logger.error(f"Training {model_type} failed: {str(e)}")
            return {'error': str(e)}
    
    def predict_with_model(self, model_type: str, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions with specific model"""
        model = self.get_model(model_type)
        if not model:
            return [{'error': f'Model type {model_type} not found'}]
        
        try:
            if model_type == 'ensemble':
                return model.predict_ensemble(texts)
            else:
                return model.predict(texts)
        except Exception as e:
            logger.error(f"Prediction with {model_type} failed: {str(e)}")
            return [{'error': str(e)}]
    
    def save_all_models(self):
        """Save all trained models"""
        for model_type, model in self.models.items():
            if model and hasattr(model, 'save_model'):
                try:
                    path = os.path.join(self.model_paths.get(model_type, 'models/'), f'{model_type}_model')
                    model.save_model(path)
                    logger.info(f"Saved {model_type} model to {path}")
                except Exception as e:
                    logger.error(f"Failed to save {model_type} model: {str(e)}")
    
    def load_all_models(self):
        """Load all saved models"""
        for model_type, model in self.models.items():
            if model and hasattr(model, 'load_model'):
                try:
                    path = os.path.join(self.model_paths.get(model_type, 'models/'), f'{model_type}_model')
                    if os.path.exists(path):
                        success = model.load_model(path)
                        if success:
                            logger.info(f"Loaded {model_type} model from {path}")
                        else:
                            logger.warning(f"Failed to load {model_type} model")
                except Exception as e:
                    logger.error(f"Error loading {model_type} model: {str(e)}")

# Initialize model manager
model_manager = ModelManager()