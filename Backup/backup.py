from flask import Flask, request, jsonify, send_from_directory, url_for, render_template
import json
import os
import traceback
import base64
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
from tqdm import tqdm
import warnings
import shutil
import pickle
import threading
import configparser
import platform
from werkzeug.utils import safe_join
import urllib.parse
import torch.nn as nn
import sys
import webbrowser
from io import BytesIO
import certifi
import ssl
import urllib.request
import urllib.error
import logging
from datetime import datetime
from pathlib import Path


# Get the user's home directory
home_dir = Path.home()

# Create a 'ijewelmatch_logs' directory in the user's documents folder
log_dir = home_dir / "Documents" / "ijewelmatch_logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Create the log file name with date
log_filename = log_dir / f"ijewelmatch_{datetime.now().strftime('%Y%m%d')}.log"

# Set up logging
logging.basicConfig(filename=str(log_filename), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a console handler for status codes
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# # Create an SSL context that uses the certifi certificate bundle
# ssl._create_default_https_context = ssl._create_unverified_context

# # Example of making an HTTPS request using the context
# url = 'https://github.com/certifi/python-certifi'
# response = urllib.request.urlopen(url, context=ssl._create_default_https_context)
# data = response.read()
# logging.debug("HTTPS request to certifi successful")
# # print(data)

# SSL Certificate Verification Setup
def setup_ssl_context():
    """
    Creates and returns a secure SSL context using certifi certificates.
    """
    try:
        # Create SSL context with certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Set verification mode to CERT_REQUIRED
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Set verification flags
        ssl_context.check_hostname = True
        
        # Set the minimum TLS version to 1.2 for security
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        logging.info("SSL context created successfully")
        return ssl_context
    except Exception as e:
        logging.error(f"Failed to create SSL context: {str(e)}")
        raise

def verify_ssl_connection(url):
    """
    Tests SSL connection to a URL using the secure context.
    
    Args:
        url (str): The HTTPS URL to test
        
    Returns:
        bool: True if connection is successful and verified
    """
    try:
        ssl_context = setup_ssl_context()
        
        # Make test request
        response = urllib.request.urlopen(url, context=ssl_context)
        
        # Log successful connection
        logging.info(f"SSL connection to {url} verified successfully")
        
        # Check certificate info
        cert = response.fp.raw._sock.getpeercert()
        logging.debug(f"Certificate details: {cert}")
        
        return True
        
    except ssl.SSLCertVerificationError as e:
        logging.error(f"Certificate verification failed for {url}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Failed to verify SSL connection to {url}: {str(e)}")
        raise

# # Test SSL connection at startup
# try:
#     test_url = 'https://github.com/certifi/python-certifi'
#     verify_ssl_connection(test_url)
#     logging.info("Initial SSL test connection successful")
# except Exception as e:
#     logging.error(f"Initial SSL test connection failed: {str(e)}")
#     # Continue running even if SSL test fails - you might want to change this behavior
#     # based on your security requirements

# Test SSL connection at startup if not explicitly skipped
skip_ssl_check = os.getenv('SKIP_SSL_CHECK', '').lower() in ('1', 'true', 'yes')

if not skip_ssl_check:
    try:
        test_url = 'https://github.com/certifi/python-certifi'
        verify_ssl_connection(test_url)
        logging.info("Initial SSL test connection successful")
    except Exception as e:
        logging.error(f"Initial SSL test connection failed: {str(e)}")
        # Continue running even if SSL test fails - you might want to change this behavior
        # based on your security requirements
else:
    logging.info("Skipping SSL verification test because SKIP_SSL_CHECK is set")

# Create default SSL context for all HTTPS requests
default_ssl_context = setup_ssl_context()
urllib.request.default_opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=default_ssl_context)
)

# Get the port from command-line arguments or default to 5002
PORT = int(sys.argv[1].split('=')[1]) if len(sys.argv) > 1 and '--port=' in sys.argv[1] else 5002

# Near the top of the file, modify:
INDEXER_STATE_FILE = os.path.join(home_dir, "Documents", "ijewelmatch_data", "base_model.pkl") 
indexer_lock = threading.Lock()

# Define the config file path
if platform.system() == "Windows":
    config_file = os.path.join(os.getenv('APPDATA'), 'imageindexer', 'config.ini')
else:  # macOS and Linux
    config_file = os.path.expanduser('~/Library/Application Support/imageindexer/config.ini')

# Ensure the directory exists
os.makedirs(os.path.dirname(config_file), exist_ok=True)

# Make sure the directory exists
os.makedirs(os.path.dirname(INDEXER_STATE_FILE), exist_ok=True)

# Create a ConfigParser object
config = configparser.ConfigParser()

# Default upload folder
UPLOAD_FOLDER = 'upload'
CURRENT_FOLDER_PATH = None

def load_config():
    global UPLOAD_FOLDER, CURRENT_FOLDER_PATH
    if os.path.exists(config_file):
        config.read(config_file)
        if 'Settings' in config:
            UPLOAD_FOLDER = config['Settings'].get('UploadFolder', UPLOAD_FOLDER)
            CURRENT_FOLDER_PATH = config['Settings'].get('CurrentFolderPath', CURRENT_FOLDER_PATH)
    else:
        # If the file doesn't exist, create it with default values
        CURRENT_FOLDER_PATH = CURRENT_FOLDER_PATH or UPLOAD_FOLDER  # Use UPLOAD_FOLDER as default if CURRENT_FOLDER_PATH is None
        config['Settings'] = {'UploadFolder': UPLOAD_FOLDER, 'CurrentFolderPath': CURRENT_FOLDER_PATH}
        save_config()

    # Ensure CURRENT_FOLDER_PATH is set
    if CURRENT_FOLDER_PATH is None:
        CURRENT_FOLDER_PATH = UPLOAD_FOLDER

    # Ensure the folder exists
    if not os.path.exists(CURRENT_FOLDER_PATH):
        os.makedirs(CURRENT_FOLDER_PATH)

def save_config():
    config['Settings'] = {'UploadFolder': UPLOAD_FOLDER, 'CurrentFolderPath': CURRENT_FOLDER_PATH}
    with open(config_file, 'w') as configfile:
        config.write(configfile)

# Call load_config at the start of your application
load_config()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

indexer = None
CURRENT_FOLDER_PATH = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress semaphore warning
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects")

app = Flask(__name__)

class JewelryMobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(JewelryMobileNetV3, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        for param in list(self.mobilenet.parameters())[:-4]:
            param.requires_grad = False
        
        # Add custom layers
        self.attention = nn.Sequential(
            nn.Conv2d(960, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        features = self.mobilenet.features(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Global average pooling
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.fc(x)
        return x

class FastImageIndexer:
#    def __init__(self, folder_path=None):
    def __init__(self, folder_path=None, weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = JewelryMobileNetV3(num_classes=960)  # Use 960 as feature dimension
        if weights_path and os.path.exists(weights_path):
            try:
                # self.model.load_state_dict(torch.load(weights_path, weights_only=True))
                state_dict = torch.load(weights_path)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                logging.error(f"Failed to load weights from {weights_path}: {e}")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.feature_dim = 960
        self.index = None
        self.image_paths = []
        
        if folder_path:
            self.create(folder_path)

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image).squeeze().cpu().numpy()
            return features / np.linalg.norm(features)
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            raise

    def create(self, folder_path):
        self.image_paths = []
        self.index = faiss.IndexFlatIP(self.feature_dim)
    
        features_list = []
        for root, dirs, files in os.walk(folder_path):
            for filename in tqdm(files):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, filename)
                    try:
                        features = self.extract_features(image_path)
                        features_list.append(features)
                        self.image_paths.append(os.path.abspath(image_path))  # Store full path
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
        
        if features_list:
            features_array = np.array(features_list).astype('float32')
            self.index.add(features_array)
        
        return len(self.image_paths)
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    def save_model_state(model, filename):
        # torch.save(model.state_dict(), filename, weights_only=True)
        torch.save(model.state_dict(), filename)

    def load_model_state(model, filename):
        # model.load_state_dict(torch.load(filename, weights_only=True))
        state_dict = torch.load(filename)
        model.load_state_dict(state_dict)
        model.eval()

    def save_state(self, filename):
        state = {
            'image_paths': self.image_paths,
            'index': faiss.serialize_index(self.index),
            'current_folder_path': CURRENT_FOLDER_PATH,
            'model_state': self.model.state_dict()
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, filename, weights_path=None):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        indexer = cls(weights_path=weights_path)
        indexer.image_paths = state['image_paths']
        indexer.index = faiss.deserialize_index(state['index'])
        global CURRENT_FOLDER_PATH
        CURRENT_FOLDER_PATH = state['current_folder_path']
        indexer.model.load_state_dict(state['model_state'])
        indexer.model.eval()
        return indexer
    
    def insert(self, image_path):
        features = self.extract_features(image_path)
        self.index.add(np.array([features]).astype('float32'))
        self.image_paths.append(os.path.abspath(image_path))
        return len(self.image_paths)

    def rebuild(self, folder_path):
        return self.create(folder_path)
    
    def search(self, query_image_path, k=5, threshold=0.8):
        query_features = self.extract_features(query_image_path)
        distances, indices = self.index.search(np.array([query_features]).astype('float32'), k)
        
        results = []
        max_similarity = max(distances[0])  # Get the maximum similarity
        for idx, dist in zip(indices[0], distances[0]):
            normalized_similarity = dist / max_similarity  # Normalize the similarity
            if normalized_similarity > threshold:
                results.append((self.image_paths[idx], float(normalized_similarity)))
        
        return results
    
    def train_model(self, train_loader, num_epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def detect_shape(image_path):
        try:
            from PIL import Image as PILImage, ImageDraw
            img = PILImage.open(image_path).convert('RGB')
            width, height = img.size
            if width > 0 and height > 0:
                return ["unidentified (shape detection requires cv2, which was removed)"]
            else:
                return ["unidentified (invalid image dimensions)"]
        except Exception as e:
            return [f"unidentified (error opening image for shape detection: {e})"]

# Initialize indexer
indexer = None
if os.path.exists(INDEXER_STATE_FILE):
    try:
        # indexer = FastImageIndexer.load_state(INDEXER_STATE_FILE)
        weights_env = os.getenv("MOBILENETV3_WEIGHTS")
        indexer = FastImageIndexer.load_state(INDEXER_STATE_FILE, weights_path=weights_env)
        print(f"Loaded existing indexer state from {INDEXER_STATE_FILE}")
    except (pickle.UnpicklingError, EOFError) as e:
        # logging.error(f"Invalid indexer state file {INDEXER_STATE_FILE}: {e}")
        # indexer = None
        logging.warning(
            f"Invalid indexer state file {INDEXER_STATE_FILE}: {e}. A new model will be generated."
        )
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            corrupt_path = f"{INDEXER_STATE_FILE}.{timestamp}.invalid"
            os.replace(INDEXER_STATE_FILE, corrupt_path)
            logging.warning(f"Renamed invalid model file to {corrupt_path}")
        except OSError as rename_err:
            logging.warning(
                f"Failed to rename invalid model file: {rename_err}. Deleting {INDEXER_STATE_FILE}."
            )
            try:
                os.remove(INDEXER_STATE_FILE)
            except OSError as remove_err:
                logging.error(
                    f"Failed to remove invalid model file {INDEXER_STATE_FILE}: {remove_err}"
                )
        weights_env = os.getenv("MOBILENETV3_WEIGHTS")
        indexer = FastImageIndexer(weights_path=weights_env)
    except Exception as e:
        print(f"Error loading indexer state: {e}")

def save_indexer_state():
    global indexer
    if indexer:
        with indexer_lock:
            indexer.save_state(INDEXER_STATE_FILE)
        app.logger.debug("Indexer state saved")

# @app.errorhandler(Exception)
# def handle_exception(e):
#     # Log the error
#     app.logger.error(f"Unhandled exception: {str(e)}")
#     app.logger.error(traceback.format_exc())
#     # Return JSON instead of HTML for HTTP errors
#     return jsonify(error=str(e)), 500

# @app.errorhandler(Exception)
# def handle_exception(e):
#     logging.error(f"Unhandled exception: {str(e)}")
#     logging.error(traceback.format_exc())
#     return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.route('/open_url')
def open_website():
    try:
        port = current_settings.get('port_number', 5002)  # Get the port from settings
        url = f"http://127.0.0.1:{port}"  # Build the URL with the correct port
        webbrowser.open_new_tab(url) 
        # return
        # webbrowser.open_new_tab(url)
        # if has_app_context():
        #     return jsonify({"message": "URL opened"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to open URL: {str(e)}"}), 500
        # logging.error(f"Failed to open URL: {e}")
        # if has_app_context():
        #     return jsonify({"error": f"Failed to open URL: {str(e)}"}), 500
    
@app.route('/')
def read_root():
    return render_template('index.html')

@app.route('/setup')
def setup():
    current_settings = {
        'port_number': config['Settings'].get('port_number', ''),
        'previous_port_numbers': config['Settings'].get('previous_port_numbers', '').split(',')
    }
    return render_template('settings.html', current_settings=current_settings)

# Load settings from a file
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'port_number': PORT, 'network_path': ''}  # Provide default values if file not found
    except json.JSONDecodeError as e:
        print(f"Error loading settings: {e}")
        return {'port_number': PORT, 'network_path': ''}  # Default on JSON error

# Save settings to a file
def save_settings(port_number, network_path):
    settings = {'port_number': port_number, 'network_path': network_path}
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

# Load current settings
current_settings = load_settings()

@app.route('/save_settings', methods=['POST'])
def save_settings_route():
    port_number = request.form.get('port_number')
    network_path = request.form.get('network_path')

    if port_number and network_path:
        try:
            port_number = int(port_number)
            save_settings(port_number, network_path)
            return jsonify({"message": "Settings saved. Please restart the application."})
        except ValueError:
            return jsonify({"error": "Invalid port number"}), 400

    return jsonify({"error": "Invalid input"}), 400

@app.route('/train', methods=['POST'])
def train():
    global indexer, CURRENT_FOLDER_PATH
    try:
        # Read the setup.json file to get network_path
        setup_file = 'setup.json'
        if os.path.exists(setup_file):
            with open(setup_file, 'r') as f:
                settings = json.load(f)
            network_path = settings.get('network_path', None)
        else:
            network_path = None
        
        # Use network_path if folder_path is not provided
        folder_path = request.json.get('folder_path', network_path)
        
        logging.info(f"Verifying folder path: {folder_path}")
        
        if folder_path is None:
            return jsonify({"error": "No folder path provided and network_path is not set"}), 400
        
        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder path does not exist"}), 400
        
        CURRENT_FOLDER_PATH = folder_path
        
        # indexer = FastImageIndexer(folder_path)
        weights_env = os.getenv("MOBILENETV3_WEIGHTS")
        indexer = FastImageIndexer(folder_path, weights_path=weights_env)
        num_images = len(indexer.image_paths)
        save_indexer_state()
        save_config()
        return jsonify({"message": f"Model trained on {num_images} images from the folder and subfolders"})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/rebuild', methods=['POST'])
def rebuild():
    global indexer
    try:
        new_folder_path = request.json['folder_path']
        if not os.path.exists(new_folder_path):
            return jsonify({"error": "New folder path does not exist"}), 400
        
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        # Rebuild the index using the new folder path
        num_new_images = indexer.rebuild(new_folder_path)
        save_indexer_state()
        return jsonify({"message": f"{num_new_images} images added to the index from the folder and subfolders"})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/insert', methods=['POST'])
def insert():
    global indexer, CURRENT_FOLDER_PATH
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        if 'image' in request.files:
            # Single image insertion
            image = request.files['image']
            if image.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            # Save the image to the current folder path
            new_path = os.path.join(CURRENT_FOLDER_PATH, image.filename)
            image.save(new_path)
            
            num_images = indexer.insert(new_path)
            save_indexer_state()
            return jsonify({"message": f"Image inserted. Total images: {num_images}"})
        
        elif 'folder_path' in request.form:
            # Folder insertion
            folder_path = request.form['folder_path']
            if not os.path.exists(folder_path):
                return jsonify({"error": "Folder path does not exist"}), 400
            
            inserted_count = 0
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, filename)
                        indexer.insert(image_path)
                        inserted_count += 1
            
            total_images = len(indexer.image_paths)
            save_indexer_state()
            return jsonify({"message": f"{inserted_count} images inserted from folder and subfolders. Total images: {total_images}"})
        
        else:
            return jsonify({"error": "No image file or folder path provided"}), 400
    
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
@app.route('/insert_folder', methods=['POST'])
def insert_folder():
    global indexer, CURRENT_FOLDER_PATH
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        folder_path = request.json.get('folder_path')
        if not folder_path:
            return jsonify({"error": "No folder path provided"}), 400
        
        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder path does not exist"}), 400
        
        inserted_count = 0
        skipped_count = 0
        errors = []

        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, filename)
                    try:
                        indexer.insert(image_path)
                        inserted_count += 1
                    except Exception as e:
                        errors.append(f"Error inserting {filename}: {str(e)}")
                else:
                    skipped_count += 1
        
        total_images = len(indexer.image_paths)
        save_indexer_state()
        
        result = {
            "message": f"{inserted_count} images inserted, {skipped_count} files skipped. Total images in index: {total_images}",
            "inserted_count": inserted_count,
            "skipped_count": skipped_count,
            "total_images": total_images
        }
        
        if errors:
            result["errors"] = errors

        return jsonify(result)
    
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route('/delete', methods=['POST'])
def delete_images():
    global indexer
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400

        filenames = request.json['filenames']
        errors = []
        successes = []

        for filename in filenames:
            full_path = next((path for path in indexer.image_paths if path.endswith(filename)), None)
            
            if full_path is None:
                errors.append(f"File {filename} not found in the index")
                continue
            
            if not os.path.exists(full_path):
                errors.append(f"File {filename} not found on disk")
                continue
            
            # Remove the file from the folder
            os.remove(full_path)
            
            # Remove the file from the indexer
            index = indexer.image_paths.index(full_path)
            indexer.image_paths.pop(index)
            # indexer.index.remove_ids(np.array([index]))
            
            # Rebuild the FAISS index since IndexFlatIP does not support removing vectors
            features_list = []
            for path in indexer.image_paths:
                try:
                    features_list.append(indexer.extract_features(path))
                except Exception as e:
                    app.logger.error(f"Error extracting features during rebuild for {path}: {e}")

            indexer.index = faiss.IndexFlatIP(indexer.feature_dim)
            if features_list:
                indexer.index.add(np.array(features_list).astype('float32'))
            
            successes.append(f"File {filename} successfully deleted")
        
        save_indexer_state()

        return jsonify({"messages": successes, "errors": errors})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    global indexer
    try:
        if indexer is None:
            # return jsonify({"error": "Model not trained yet. Use /train first."}), 400
            app.logger.error("Search attempted but indexer is None")
            return jsonify({"error": "Indexer not initialized"}), 400

        if getattr(indexer, 'index', None) is None:
            app.logger.error("Search attempted but indexer.index is None")
            return jsonify({"error": "Indexer not initialized"}), 400

        if not indexer.image_paths:
            app.logger.error("Search attempted but no images indexed")
            return jsonify({"error": "No images indexed"}), 400
        
        # Check if the request has the correct content type
        if request.content_type.startswith('application/json'):
            data = request.json
        elif request.content_type.startswith('multipart/form-data'):
            data = request.form
        else:
            return jsonify({"error": f"Unsupported content type: {request.content_type}"}), 400

        # Handle image data
        if 'image' in request.files:
            image = request.files['image']
            if image.filename == '':
                return jsonify({"error": "No selected file"}), 400
            image_data = image.read()
        elif 'image' in data:
            # Handle base64 encoded image
            try:
                image_data = base64.b64decode(data['image'].split(',')[1])
            except:
                return jsonify({"error": "Invalid base64 image data"}), 400
        else:
            return jsonify({"error": "No image file or data provided"}), 400
        
        # Ensure UPLOAD_FOLDER exists and use absolute path
        upload_folder = os.path.abspath(UPLOAD_FOLDER)
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder, exist_ok=True)
        
        # Save the image temporarily with proper permissions
        temp_image_path = os.path.join(upload_folder, 'temp_search_image.jpg')
        try:
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
        except PermissionError:
            # If we can't write to upload folder, try temp directory
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_image_path = os.path.join(temp_dir, 'temp_search_image.jpg')
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
        
        # Get Number_Of_Images_Req and Similarity_Percentage values
        Number_Of_Images_Req = int(data.get('Number_Of_Images_Req', 5))
        Similarity_Percentage_str = data.get('Similarity_Percentage')
        if Similarity_Percentage_str and str(Similarity_Percentage_str).strip():
            Similarity_Percentage = float(Similarity_Percentage_str) / 100
            if Similarity_Percentage == 1.0:
                Similarity_Percentage = 0.9999  # Use a very high threshold instead of 1.0
        else:
            Similarity_Percentage = 0.8  # Default similarity percentage

        results = indexer.search(temp_image_path, k=Number_Of_Images_Req, threshold=Similarity_Percentage)
        # print("Search results:", results)  # Debugging line
        
        # Remove the temporary image
        try:
            os.remove(temp_image_path)
        except:
            pass  # Ignore errors when cleaning up temp file

        # Convert file paths to URLs
        results_with_urls = [
            {
                "image_name": os.path.basename(filename),
                "url": url_for('serve_image', filename=os.path.basename(filename), _external=True),
                "similarity": float(sim)
            } for filename, sim in results
        ]

        return jsonify({
            "results": results_with_urls,
            "Status_Code": 200,
            "Response": "Success!"
        })
        
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        
@app.route('/uploads/<path:filename>')
def serve_image(filename):
    try:
        app.logger.debug(f"Attempting to serve image: {filename}")
        
        # Find the full path in indexer.image_paths
        full_path = next((path for path in indexer.image_paths if os.path.basename(path) == filename), None)
        
        if full_path is None or not os.path.exists(full_path):
            app.logger.error(f"File not found: {filename}")
            return jsonify({"error": f"Image {filename} not found"}), 404


        app.logger.debug(f"Full path to image: {full_path}")
        app.logger.debug(f"File exists: {os.path.exists(full_path)}")

        if not os.path.exists(full_path):
            app.logger.error(f"File not found: {full_path}")
            return jsonify({"error": f"Image {filename} not found"}), 404

        # Get the directory and filename separately
        directory = os.path.dirname(full_path)
        basename = os.path.basename(full_path)

        return send_from_directory(directory, basename)
    except Exception as e:
        app.logger.error(f"Error serving image {filename}: {str(e)}")
        return jsonify({"error": f"Error serving image: {str(e)}"}), 500
    
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/config', methods=['GET', 'POST'])
def config_route():
    global CURRENT_FOLDER_PATH, indexer
    if request.method == 'GET':
        return jsonify({"current_folder": CURRENT_FOLDER_PATH})
    elif request.method == 'POST':
        new_folder_path = request.json.get('folder_path')
        if new_folder_path:
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            CURRENT_FOLDER_PATH = new_folder_path
            save_config()
            if indexer:
                save_indexer_state()
            return jsonify({"message": "Folder path updated successfully", "new_folder_path": CURRENT_FOLDER_PATH})
        else:
            return jsonify({"error": "No folder_path provided in request"}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    logging.error(traceback.format_exc())
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == '__main__':
    port_number = current_settings.get('port_number', 5002)
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    open_website()
    logging.info(f"Starting server on port {port_number}")
    app.run(debug=False, port=int(port_number), host='0.0.0.0')










# 22/08/2025

from flask import Flask, request, jsonify, send_from_directory, url_for, render_template
import json
import os
import traceback
import base64
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
from tqdm import tqdm
import warnings
import shutil
import pickle
import threading
import configparser
import platform
from werkzeug.utils import safe_join
import urllib.parse
import torch.nn as nn
import sys
import webbrowser
from io import BytesIO
import certifi
import ssl
import urllib.request
import logging
from datetime import datetime
from pathlib import Path

# Get the user's home directory
home_dir = Path.home()

# Create a 'ijewelmatch_logs' directory in the user's documents folder
log_dir = home_dir / "Documents" / "ijewelmatch_logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Create the log file name with date
log_filename = log_dir / f"ijewelmatch_{datetime.now().strftime('%Y%m%d')}.log"

# Set up logging
logging.basicConfig(filename=str(log_filename), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a console handler for status codes
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# # Create an SSL context that uses the certifi certificate bundle
# ssl._create_default_https_context = ssl._create_unverified_context

# # Example of making an HTTPS request using the context
# url = 'https://github.com/certifi/python-certifi'
# response = urllib.request.urlopen(url, context=ssl._create_default_https_context)
# data = response.read()
# logging.debug("HTTPS request to certifi successful")
# # print(data)

# SSL Certificate Verification Setup
def setup_ssl_context():
    """
    Creates and returns a secure SSL context using certifi certificates.
    """
    try:
        # Create SSL context with certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Set verification mode to CERT_REQUIRED
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Set verification flags
        ssl_context.check_hostname = True
        
        # Set the minimum TLS version to 1.2 for security
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        logging.info("SSL context created successfully")
        return ssl_context
    except Exception as e:
        logging.error(f"Failed to create SSL context: {str(e)}")
        raise

def verify_ssl_connection(url):
    """
    Tests SSL connection to a URL using the secure context.
    
    Args:
        url (str): The HTTPS URL to test
        
    Returns:
        bool: True if connection is successful and verified
    """
    try:
        ssl_context = setup_ssl_context()
        
        # Make test request
        response = urllib.request.urlopen(url, context=ssl_context)
        
        # Log successful connection
        logging.info(f"SSL connection to {url} verified successfully")
        
        # Check certificate info
        cert = response.fp.raw._sock.getpeercert()
        logging.debug(f"Certificate details: {cert}")
        
        return True
        
    except ssl.SSLCertVerificationError as e:
        logging.error(f"Certificate verification failed for {url}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Failed to verify SSL connection to {url}: {str(e)}")
        raise

# # Test SSL connection at startup
# try:
#     test_url = 'https://github.com/certifi/python-certifi'
#     verify_ssl_connection(test_url)
#     logging.info("Initial SSL test connection successful")
# except Exception as e:
#     logging.error(f"Initial SSL test connection failed: {str(e)}")
#     # Continue running even if SSL test fails - you might want to change this behavior
#     # based on your security requirements

# # Create default SSL context for all HTTPS requests
# default_ssl_context = setup_ssl_context()
# urllib.request.default_opener = urllib.request.build_opener(
#     urllib.request.HTTPSHandler(context=default_ssl_context)
# )

# Test SSL connection at startup if not explicitly skipped
skip_ssl_check = os.getenv('SKIP_SSL_CHECK', '').lower() in ('1', 'true', 'yes')

if not skip_ssl_check:
    try:
        test_url = 'https://github.com/certifi/python-certifi'
        verify_ssl_connection(test_url)
        logging.info("Initial SSL test connection successful")
    except Exception as e:
        logging.error(f"Initial SSL test connection failed: {str(e)}")
        # Continue running even if SSL test fails - you might want to change this behavior
        # based on your security requirements
else:
    logging.info("Skipping SSL verification test because SKIP_SSL_CHECK is set")

# Create default SSL context for all HTTPS requests
default_ssl_context = setup_ssl_context()
urllib.request.default_opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=default_ssl_context)
)

# Get the port from command-line arguments or default to 5002
PORT = int(sys.argv[1].split('=')[1]) if len(sys.argv) > 1 and '--port=' in sys.argv[1] else 5002

# Near the top of the file, modify:
INDEXER_STATE_FILE = os.path.join(home_dir, "Documents", "ijewelmatch_data", "base_model.pkl")
indexer_lock = threading.Lock()

# Define the config file path
if platform.system() == "Windows":
    config_file = os.path.join(os.getenv('APPDATA'), 'imageindexer', 'config.ini')
else:  # macOS and Linux
    config_file = os.path.expanduser('~/Library/Application Support/imageindexer/config.ini')

# Ensure the directory exists
os.makedirs(os.path.dirname(config_file), exist_ok=True)

# Make sure the directory exists
os.makedirs(os.path.dirname(INDEXER_STATE_FILE), exist_ok=True)

# Create a ConfigParser object
config = configparser.ConfigParser()

# Default upload folder
UPLOAD_FOLDER = 'upload'
CURRENT_FOLDER_PATH = None

def load_config():
    global UPLOAD_FOLDER, CURRENT_FOLDER_PATH
    if os.path.exists(config_file):
        config.read(config_file)
        if 'Settings' in config:
            UPLOAD_FOLDER = config['Settings'].get('UploadFolder', UPLOAD_FOLDER)
            CURRENT_FOLDER_PATH = config['Settings'].get('CurrentFolderPath', CURRENT_FOLDER_PATH)
    else:
        # If the file doesn't exist, create it with default values
        CURRENT_FOLDER_PATH = CURRENT_FOLDER_PATH or UPLOAD_FOLDER  # Use UPLOAD_FOLDER as default if CURRENT_FOLDER_PATH is None
        config['Settings'] = {'UploadFolder': UPLOAD_FOLDER, 'CurrentFolderPath': CURRENT_FOLDER_PATH}
        save_config()

    # Ensure CURRENT_FOLDER_PATH is set
    if CURRENT_FOLDER_PATH is None:
        CURRENT_FOLDER_PATH = UPLOAD_FOLDER

    # Ensure the folder exists
    if not os.path.exists(CURRENT_FOLDER_PATH):
        os.makedirs(CURRENT_FOLDER_PATH)

def save_config():
    config['Settings'] = {'UploadFolder': UPLOAD_FOLDER, 'CurrentFolderPath': CURRENT_FOLDER_PATH}
    with open(config_file, 'w') as configfile:
        config.write(configfile)

# Call load_config at the start of your application
load_config()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

indexer = None
CURRENT_FOLDER_PATH = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress semaphore warning
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects")

app = Flask(__name__)

# class JewelryMobileNetV3(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(JewelryMobileNetV3, self).__init__()
#         self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
#         # Freeze early layers
#         for param in list(self.mobilenet.parameters())[:-4]:
#             param.requires_grad = False
        
#         # Add custom layers
#         self.attention = nn.Sequential(
#             nn.Conv2d(960, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(960, num_classes)

#     def forward(self, x):
#         features = self.mobilenet.features(x)
        
#         # Apply attention
#         attention_weights = self.attention(features)
#         features = features * attention_weights
        
#         # Global average pooling
#         x = self.global_pool(features)
#         x = x.view(x.size(0), -1)
        
#         # Classification layer
#         x = self.fc(x)
#         return x

class JewelryMobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(JewelryMobileNetV3, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        for param in list(self.mobilenet.parameters())[:-4]:
            param.requires_grad = False
        
        # Add custom layers
        self.attention = nn.Sequential(
            nn.Conv2d(960, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        features = self.mobilenet.features(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Global average pooling
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.fc(x)
        return x

class FastImageIndexer:
    def __init__(self, folder_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = JewelryMobileNetV3(num_classes=960)  # Use 960 as feature dimension
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.feature_dim = 960
        self.index = None
        self.image_paths = []
        
        if folder_path:
            self.create(folder_path)

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image).squeeze().cpu().numpy()
            return features / np.linalg.norm(features)
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            raise

    def create(self, folder_path):
        self.image_paths = []
        self.index = faiss.IndexFlatIP(self.feature_dim)
    
        features_list = []
        for root, dirs, files in os.walk(folder_path):
            for filename in tqdm(files):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, filename)
                    try:
                        features = self.extract_features(image_path)
                        features_list.append(features)
                        self.image_paths.append(os.path.abspath(image_path))  # Store full path
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
        
        if features_list:
            features_array = np.array(features_list).astype('float32')
            self.index.add(features_array)
        
        return len(self.image_paths)
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    def save_model_state(model, filename):
        torch.save(model.state_dict(), filename, weights_only=True)

    def load_model_state(model, filename):
        model.load_state_dict(torch.load(filename, weights_only=True))
        model.eval()

    def save_state(self, filename):
        state = {
            'image_paths': self.image_paths,
            'index': faiss.serialize_index(self.index),
            'current_folder_path': CURRENT_FOLDER_PATH,
            'model_state': self.model.state_dict()
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        indexer = cls()
        indexer.image_paths = state['image_paths']
        indexer.index = faiss.deserialize_index(state['index'])
        global CURRENT_FOLDER_PATH
        CURRENT_FOLDER_PATH = state['current_folder_path']
        indexer.model.load_state_dict(state['model_state'])
        indexer.model.eval()
        return indexer
    
    def insert(self, image_path):
        features = self.extract_features(image_path)
        self.index.add(np.array([features]).astype('float32'))
        self.image_paths.append(os.path.abspath(image_path))
        return len(self.image_paths)

    def rebuild(self, folder_path):
        return self.create(folder_path)
    
    def search(self, query_image_path, k=5, threshold=0.8):
        query_features = self.extract_features(query_image_path)
        distances, indices = self.index.search(np.array([query_features]).astype('float32'), k)
        
        results = []
        max_similarity = max(distances[0])  # Get the maximum similarity
        for idx, dist in zip(indices[0], distances[0]):
            normalized_similarity = dist / max_similarity  # Normalize the similarity
            if normalized_similarity > threshold:
                results.append((self.image_paths[idx], float(normalized_similarity)))
        
        return results
    
    def train_model(self, train_loader, num_epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def detect_shape(image_path):
        try:
            from PIL import Image as PILImage, ImageDraw
            img = PILImage.open(image_path).convert('RGB')
            width, height = img.size
            if width > 0 and height > 0:
                return ["unidentified (shape detection requires cv2, which was removed)"]
            else:
                return ["unidentified (invalid image dimensions)"]
        except Exception as e:
            return [f"unidentified (error opening image for shape detection: {e})"]

# Initialize indexer
indexer = None
if os.path.exists(INDEXER_STATE_FILE):
    try:
        indexer = FastImageIndexer.load_state(INDEXER_STATE_FILE)
        print(f"Loaded existing indexer state from {INDEXER_STATE_FILE}")
    except Exception as e:
        print(f"Error loading indexer state: {e}")

def save_indexer_state():
    global indexer
    if indexer:
        with indexer_lock:
            indexer.save_state(INDEXER_STATE_FILE)
        app.logger.debug("Indexer state saved")

# @app.errorhandler(Exception)
# def handle_exception(e):
#     # Log the error
#     app.logger.error(f"Unhandled exception: {str(e)}")
#     app.logger.error(traceback.format_exc())
#     # Return JSON instead of HTML for HTTP errors
#     return jsonify(error=str(e)), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    logging.error(traceback.format_exc())
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.route('/open_url')  
def open_website():
    try:
        port = current_settings.get('port_number', 5002) # Get the port from settings
        url = f"http://127.0.0.1:{port}"  # Build the URL with the correct port
        webbrowser.open_new_tab(url) 
        # return
    except Exception as e:
        return jsonify({"error": f"Failed to open URL: {str(e)}"}), 500
    
@app.route('/')
def read_root():
    return render_template('index.html')

@app.route('/setup')
def setup():
    current_settings = {
        'port_number': config['Settings'].get('port_number', ''),
        'previous_port_numbers': config['Settings'].get('previous_port_numbers', '').split(',')
    }
    return render_template('settings.html', current_settings=current_settings)

# Load settings from a file
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'port_number': PORT, 'network_path': ''}  # Provide default values if file not found
    except json.JSONDecodeError as e:
        print(f"Error loading settings: {e}")
        return {'port_number': PORT, 'network_path': ''}  # Default on JSON error

# Save settings to a file
def save_settings(port_number, network_path):
    settings = {'port_number': port_number, 'network_path': network_path}
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

# Load current settings
current_settings = load_settings()

@app.route('/save_settings', methods=['POST'])
def save_settings_route():
    port_number = request.form.get('port_number')
    network_path = request.form.get('network_path')

    if port_number and network_path:
        try:
            port_number = int(port_number)
            save_settings(port_number, network_path)
            return jsonify({"message": "Settings saved. Please restart the application."})
        except ValueError:
            return jsonify({"error": "Invalid port number"}), 400

    return jsonify({"error": "Invalid input"}), 400

@app.route('/train', methods=['POST'])
def train():
    global indexer, CURRENT_FOLDER_PATH
    try:
        # Read the setup.json file to get network_path
        setup_file = 'setup.json'
        if os.path.exists(setup_file):
            with open(setup_file, 'r') as f:
                settings = json.load(f)
            network_path = settings.get('network_path', None)
        else:
            network_path = None
        
        # Use network_path if folder_path is not provided
        folder_path = request.json.get('folder_path', network_path)
        
        if folder_path is None:
            return jsonify({"error": "No folder path provided and network_path is not set"}), 400
        
        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder path does not exist"}), 400
        
        CURRENT_FOLDER_PATH = folder_path
        
        indexer = FastImageIndexer(folder_path)
        num_images = len(indexer.image_paths)
        save_indexer_state()
        save_config()
        return jsonify({"message": f"Model trained on {num_images} images from the folder and subfolders"})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/rebuild', methods=['POST'])
def rebuild():
    global indexer
    try:
        new_folder_path = request.json['folder_path']
        if not os.path.exists(new_folder_path):
            return jsonify({"error": "New folder path does not exist"}), 400
        
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        # Rebuild the index using the new folder path
        num_new_images = indexer.rebuild(new_folder_path)
        save_indexer_state()
        return jsonify({"message": f"{num_new_images} images added to the index from the folder and subfolders"})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/insert', methods=['POST'])
def insert():
    global indexer, CURRENT_FOLDER_PATH
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        if 'image' in request.files:
            # Single image insertion
            image = request.files['image']
            if image.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            # Save the image to the current folder path
            new_path = os.path.join(CURRENT_FOLDER_PATH, image.filename)
            image.save(new_path)
            
            num_images = indexer.insert(new_path)
            save_indexer_state()
            return jsonify({"message": f"Image inserted. Total images: {num_images}"})
        
        elif 'folder_path' in request.form:
            # Folder insertion
            folder_path = request.form['folder_path']
            if not os.path.exists(folder_path):
                return jsonify({"error": "Folder path does not exist"}), 400
            
            inserted_count = 0
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, filename)
                        indexer.insert(image_path)
                        inserted_count += 1
            
            total_images = len(indexer.image_paths)
            save_indexer_state()
            return jsonify({"message": f"{inserted_count} images inserted from folder and subfolders. Total images: {total_images}"})
        
        else:
            return jsonify({"error": "No image file or folder path provided"}), 400
    
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
@app.route('/insert_folder', methods=['POST'])
def insert_folder():
    global indexer, CURRENT_FOLDER_PATH
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        folder_path = request.json.get('folder_path')
        if not folder_path:
            return jsonify({"error": "No folder path provided"}), 400
        
        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder path does not exist"}), 400
        
        inserted_count = 0
        skipped_count = 0
        errors = []

        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, filename)
                    try:
                        indexer.insert(image_path)
                        inserted_count += 1
                    except Exception as e:
                        errors.append(f"Error inserting {filename}: {str(e)}")
                else:
                    skipped_count += 1
        
        total_images = len(indexer.image_paths)
        save_indexer_state()
        
        result = {
            "message": f"{inserted_count} images inserted, {skipped_count} files skipped. Total images in index: {total_images}",
            "inserted_count": inserted_count,
            "skipped_count": skipped_count,
            "total_images": total_images
        }
        
        if errors:
            result["errors"] = errors

        return jsonify(result)
    
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route('/delete', methods=['POST'])
def delete_images():
    global indexer
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400

        filenames = request.json['filenames']
        errors = []
        successes = []

        for filename in filenames:
            full_path = next((path for path in indexer.image_paths if path.endswith(filename)), None)
            
            if full_path is None:
                errors.append(f"File {filename} not found in the index")
                continue
            
            if not os.path.exists(full_path):
                errors.append(f"File {filename} not found on disk")
                continue
            
            # Remove the file from the folder
            os.remove(full_path)
            
            # Remove the file from the indexer
            index = indexer.image_paths.index(full_path)
            indexer.image_paths.pop(index)
            indexer.index.remove_ids(np.array([index]))
            
            successes.append(f"File {filename} successfully deleted")
        
        save_indexer_state()

        return jsonify({"messages": successes, "errors": errors})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    global indexer
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        # Check if the request has the correct content type
        if request.content_type.startswith('application/json'):
            data = request.json
        elif request.content_type.startswith('multipart/form-data'):
            data = request.form
        else:
            return jsonify({"error": f"Unsupported content type: {request.content_type}"}), 400

        # Handle image data
        if 'image' in request.files:
            image = request.files['image']
            if image.filename == '':
                return jsonify({"error": "No selected file"}), 400
            image_data = image.read()
        elif 'image' in data:
            # Handle base64 encoded image
            try:
                image_data = base64.b64decode(data['image'].split(',')[1])
            except:
                return jsonify({"error": "Invalid base64 image data"}), 400
        else:
            return jsonify({"error": "No image file or data provided"}), 400
        
        # Ensure UPLOAD_FOLDER exists and use absolute path
        upload_folder = os.path.abspath(UPLOAD_FOLDER)
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder, exist_ok=True)
        
        # Save the image temporarily with proper permissions
        temp_image_path = os.path.join(upload_folder, 'temp_search_image.jpg')
        try:
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
        except PermissionError:
            # If we can't write to upload folder, try temp directory
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_image_path = os.path.join(temp_dir, 'temp_search_image.jpg')
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
        
        # Get Number_Of_Images_Req and Similarity_Percentage values
        Number_Of_Images_Req = int(data.get('Number_Of_Images_Req', 5))
        Similarity_Percentage_str = data.get('Similarity_Percentage')
        if Similarity_Percentage_str and str(Similarity_Percentage_str).strip():
            Similarity_Percentage = float(Similarity_Percentage_str) / 100
            if Similarity_Percentage == 1.0:
                Similarity_Percentage = 0.9999  # Use a very high threshold instead of 1.0
        else:
            Similarity_Percentage = 0.8  # Default similarity percentage

        results = indexer.search(temp_image_path, k=Number_Of_Images_Req, threshold=Similarity_Percentage)
        # print("Search results:", results)  # Debugging line
        
        # Remove the temporary image
        try:
            os.remove(temp_image_path)
        except:
            pass  # Ignore errors when cleaning up temp file

        # Convert file paths to URLs
        results_with_urls = [
            {
                "image_name": os.path.basename(filename),
                "url": url_for('serve_image', filename=os.path.basename(filename), _external=True),
                "similarity": float(sim)
            } for filename, sim in results
        ]

        return jsonify({
            "results": results_with_urls,
            "Status_Code": 200,
            "Response": "Success!"
        })
        
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        
@app.route('/uploads/<path:filename>')
def serve_image(filename):
    try:
        app.logger.debug(f"Attempting to serve image: {filename}")
        
        # Find the full path in indexer.image_paths
        full_path = next((path for path in indexer.image_paths if os.path.basename(path) == filename), None)
        
        if full_path is None or not os.path.exists(full_path):
            app.logger.error(f"File not found: {filename}")
            return jsonify({"error": f"Image {filename} not found"}), 404


        app.logger.debug(f"Full path to image: {full_path}")
        app.logger.debug(f"File exists: {os.path.exists(full_path)}")

        if not os.path.exists(full_path):
            app.logger.error(f"File not found: {full_path}")
            return jsonify({"error": f"Image {filename} not found"}), 404

        # Get the directory and filename separately
        directory = os.path.dirname(full_path)
        basename = os.path.basename(full_path)

        return send_from_directory(directory, basename)
    except Exception as e:
        app.logger.error(f"Error serving image {filename}: {str(e)}")
        return jsonify({"error": f"Error serving image: {str(e)}"}), 500
    
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/config', methods=['GET', 'POST'])
def config_route():
    global CURRENT_FOLDER_PATH, indexer
    if request.method == 'GET':
        return jsonify({"current_folder": CURRENT_FOLDER_PATH})
    elif request.method == 'POST':
        new_folder_path = request.json.get('folder_path')
        if new_folder_path:
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            CURRENT_FOLDER_PATH = new_folder_path
            save_config()
            if indexer:
                save_indexer_state()
            return jsonify({"message": "Folder path updated successfully", "new_folder_path": CURRENT_FOLDER_PATH})
        else:
            return jsonify({"error": "No folder_path provided in request"}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    logging.error(traceback.format_exc())
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == '__main__':
    port_number = current_settings.get('port_number', 5002)
    open_website()
    logging.info(f"Starting server on port {port_number}")
    app.run(debug=False, port=int(port_number), host='0.0.0.0')
