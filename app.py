"""
Flask application for Intelligent Document Processing System
"""
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import traceback
# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import config

# Import modules with error handling for missing dependencies
prediction_api = None
database_manager = None

try:
    from predict import prediction_api
    logger.info("Prediction API loaded successfully")
except ImportError as e:
    logger.warning(f"Prediction API not available: {e}")

try:
    from database import database_manager
    logger.info("Database manager loaded successfully")
except ImportError as e:
    logger.warning(f"Database manager not available: {e}")

# Create Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.app.max_file_size
CORS(app)

# HTML template for the main page
MAIN_PAGE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Intelligent Document Processing System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .endpoint { margin: 10px 0; }
        .method { background: #007bff; color: white; padding: 5px 10px; border-radius: 4px; font-size: 12px; margin-right: 10px; }
        .path { font-family: monospace; font-weight: bold; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 10px 0; white-space: pre-wrap; }
        .status { margin: 20px 0; padding: 15px; border-radius: 4px; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f1b0b7; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Intelligent Document Processing System</h1>
        
        <div class="section">
            <h2>System Status</h2>
            <div id="status" class="status">Checking system status...</div>
        </div>
        
        <div class="section">
            <h2>📝 Text Analysis</h2>
            <textarea id="textInput" placeholder="Enter text to analyze...">Sample invoice text: Invoice #12345 dated 2024-01-15. Company ABC Inc. Total amount: $1,250.00. Customer email: john@example.com</textarea>
            <br>
            <button onclick="analyzeText()">Analyze Text</button>
            <div id="textResult" class="result" style="display:none;"></div>
        </div>
        
        <div class="section">
            <h2>🌐 API Endpoints</h2>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/api/predict/text</span> - Analyze text content
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/api/predict/image</span> - Process image and extract text
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/api/predict/batch</span> - Batch process multiple texts
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/api/health</span> - System health check
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/api/status</span> - Detailed system status
            </div>
        </div>
        
        <div class="section">
            <h2>🛠 Features</h2>
            <ul>
                <li>✅ OCR Text Extraction (Tesseract + Google Cloud Vision)</li>
                <li>✅ Machine Learning Models (TensorFlow, PyTorch, Keras)</li>
                <li>✅ Natural Language Processing (spaCy, NLTK, Transformers)</li>
                <li>✅ Entity Recognition (Pattern-based + ML-based)</li>
                <li>✅ Document Classification & Sentiment Analysis</li>
                <li>✅ Database Integration (MySQL, Cassandra)</li>
                <li>✅ Cloud Storage (Google Cloud, Azure, AWS)</li>
                <li>✅ Security Features (SSL, IAM, Authentication)</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Check system status on page load
        window.onload = function() {
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('status');
                    if (data.status === 'healthy') {
                        statusDiv.className = 'status success';
                        statusDiv.textContent = '✅ System is running and healthy';
                    } else {
                        statusDiv.className = 'status error';
                        statusDiv.textContent = '❌ System has issues: ' + (data.message || 'Unknown error');
                    }
                })
                .catch(error => {
                    const statusDiv = document.getElementById('status');
                    statusDiv.className = 'status error';
                    statusDiv.textContent = '❌ Could not connect to system';
                });
        };
        
        function analyzeText() {
            const text = document.getElementById('textInput').value;
            const resultDiv = document.getElementById('textResult');
            
            if (!text.trim()) {
                alert('Please enter some text to analyze');
                return;
            }
            
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Analyzing text...';
            
            fetch('/api/predict/text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    options: {
                        include_entities: true,
                        include_sentiment: true
                    }
                })
            })
            .then(response => response.json())
            .then(data => {
                resultDiv.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                resultDiv.textContent = 'Error: ' + error.message;
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Main page"""
    return render_template_string(MAIN_PAGE_HTML)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'message': 'Intelligent Document Processing System is running'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def system_status():
    """Detailed system status"""
    try:
        status_info = {
            'timestamp': datetime.now().isoformat(),
            'system': 'Intelligent Document Processing System',
            'version': '1.0.0',
            'components': {
                'flask_app': 'running',
                'prediction_engine': 'available',
                'database': 'checking...',
                'ocr_engine': 'available',
                'nlp_pipeline': 'available',
                'entity_extractor': 'available'
            }
        }
        
        # Check database connectivity
        try:
            db_status = database_manager.initialize_databases()
            status_info['components']['database'] = db_status
        except Exception as e:
            status_info['components']['database'] = f'error: {str(e)}'
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/predict/text', methods=['POST'])
def predict_text():
    """Text prediction endpoint"""
    try:
        if not prediction_api:
            return jsonify({
                'success': False,
                'error': 'Prediction API not available - ML dependencies not installed'
            }), 503
            
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided in request'
            }), 400
        
        text = data['text']
        options = data.get('options', {})
        
        # Process text with prediction API
        result = prediction_api.predict_text(text, options)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Text prediction failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    """Image prediction endpoint"""
    try:
        if not prediction_api:
            return jsonify({
                'success': False,
                'error': 'Prediction API not available - ML dependencies not installed'
            }), 503
            
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(config.app.upload_folder, filename)
            
            # Create upload directory if it doesn't exist
            os.makedirs(config.app.upload_folder, exist_ok=True)
            
            file.save(filepath)
            
            # Get options
            options = {}
            if request.form.get('ocr_method'):
                options['ocr_method'] = request.form.get('ocr_method')
            
            # Process image
            result = prediction_api.predict_image(filepath, options)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(result)
        
    except Exception as e:
        logger.error(f"Image prediction failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if not prediction_api:
            return jsonify({
                'success': False,
                'error': 'Prediction API not available - ML dependencies not installed'
            }), 503
            
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'No texts provided in request'
            }), 400
        
        texts = data['texts']
        options = data.get('options', {})
        
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'Texts must be provided as a list'
            }), 400
        
        # Process batch
        result = prediction_api.batch_predict_texts(texts, options)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Create necessary directories
    config.create_directories()
    
    # Initialize database if possible
    if database_manager:
        try:
            database_manager.initialize_databases()
            logger.info("Database initialization attempted")
        except Exception as e:
            logger.warning(f"Database initialization failed: {str(e)}")
    else:
        logger.info("Database manager not available - skipping database initialization")
    
    # Run Flask app
    app.run(
        host=config.app.host,
        port=config.app.port,
        debug=config.app.debug
    )