from flask import Flask, request, jsonify, send_from_directory, url_for, render_template
import json
import os
import traceback
import base64
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# =============================
# Logging setup
# =============================
home_dir = Path.home()
log_dir = home_dir / "Documents" / "ijewelmatch_logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f"ijewelmatch_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(filename=str(log_filename), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# =============================
# SSL helpers
# =============================
def setup_ssl_context():
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.check_hostname = True
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        logging.info("SSL context created successfully")
        return ssl_context
    except Exception as e:
        logging.error(f"Failed to create SSL context: {str(e)}")
        raise

def verify_ssl_connection(url):
    try:
        ssl_context = setup_ssl_context()
        response = urllib.request.urlopen(url, context=ssl_context)
        logging.info(f"SSL connection to {url} verified successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to verify SSL connection to {url}: {str(e)}")
        raise

skip_ssl_check = os.getenv('SKIP_SSL_CHECK', '').lower() in ('1', 'true', 'yes')
if not skip_ssl_check:
    try:
        verify_ssl_connection('https://github.com/certifi/python-certifi')
        logging.info("Initial SSL test connection successful")
    except Exception as e:
        logging.error(f"Initial SSL test connection failed: {str(e)}")
else:
    logging.info("Skipping SSL verification test because SKIP_SSL_CHECK is set")

default_ssl_context = setup_ssl_context()
_opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=default_ssl_context)
)
urllib.request.install_opener(_opener)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()

# =============================
# Port & paths
# =============================
PORT = int(sys.argv[1].split('=')[1]) if len(sys.argv) > 1 and '--port=' in sys.argv[1] else 5002
INDEXER_STATE_FILE = os.path.join(str(home_dir), "Documents", "ijewelmatch_data", "base_model.pkl")
indexer_lock = threading.Lock()

# Config path with Windows fallback
if platform.system() == "Windows":
    appdata_dir = os.getenv('APPDATA')
    if not appdata_dir:
        # Fallback if APPDATA is not set (common on some Windows Server envs)
        appdata_dir = os.path.join(str(Path.home()), "AppData", "Roaming")
    config_file = os.path.join(appdata_dir, 'imageindexer', 'config.ini')
else:
    config_file = os.path.expanduser('~/Library/Application Support/imageindexer/config.ini')

os.makedirs(os.path.dirname(config_file), exist_ok=True)
os.makedirs(os.path.dirname(INDEXER_STATE_FILE), exist_ok=True)

config = configparser.ConfigParser()
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
        CURRENT_FOLDER_PATH = CURRENT_FOLDER_PATH or UPLOAD_FOLDER
        config['Settings'] = {'UploadFolder': UPLOAD_FOLDER, 'CurrentFolderPath': CURRENT_FOLDER_PATH}
        save_config()

    if CURRENT_FOLDER_PATH is None:
        CURRENT_FOLDER_PATH = UPLOAD_FOLDER
    if not os.path.exists(CURRENT_FOLDER_PATH):
        os.makedirs(CURRENT_FOLDER_PATH)


def save_config():
    config['Settings'] = {'UploadFolder': UPLOAD_FOLDER, 'CurrentFolderPath': CURRENT_FOLDER_PATH}
    with open(config_file, 'w') as configfile:
        config.write(configfile)


load_config()
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# =============================
# Model definition
# =============================
class JewelryMobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(JewelryMobileNetV3, self).__init__()
        try:
            self.mobilenet = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
            logging.info("MobileNetV3 weights loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load pretrained weights ({e}). Falling back to random weights.")
            self.mobilenet = models.mobilenet_v3_large(weights=None)

        for param in list(self.mobilenet.parameters())[:-4]:
            param.requires_grad = False

        self.attention = nn.Sequential(
            nn.Conv2d(960, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        features = self.mobilenet.features(x)
        attention_weights = self.attention(features)
        features = features * attention_weights
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =============================
# Indexer (multi-threaded)
# =============================
class FastImageIndexer:
    def __init__(self, folder_path=None, max_workers=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = JewelryMobileNetV3(num_classes=960).to(self.device)
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

        # default workers: CPU count or env override
        cpu_cnt = os.cpu_count() or 4
        self.max_workers = int(max_workers or os.getenv('IJM_MAX_WORKERS', max(4, min(32, cpu_cnt))))

        if folder_path:
            self.create(folder_path)

    # --- robust feature extraction ---
    def extract_features(self, image_path):
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
        except UnidentifiedImageError:
            print(f"Skipped invalid/corrupt image (Unidentified): {image_path}")
            return None
        except Exception as e:
            print(f"Skipped unreadable image {image_path}: {e}")
            return None

        try:
            image = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image).squeeze().cpu().numpy()
            norm = np.linalg.norm(features)
            if norm == 0 or not np.isfinite(norm):
                print(f"Skipped image with zero/NaN features: {image_path}")
                return None
            return features / norm
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return None

    # --- multi-threaded create ---
    def create(self, folder_path):
        self.image_paths = []
        self.index = faiss.IndexFlatIP(self.feature_dim)

        valid_exts = ('.png', '.jpg', '.jpeg', '.jfif')
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(valid_exts):
                    image_files.append(os.path.join(root, filename))

        features_list = []
        inserted, skipped = 0, 0
        workers = max(1, self.max_workers)
        logging.info(f"Building index with {workers} workers over {len(image_files)} images…")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.extract_features, p): p for p in image_files}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                path = futures[fut]
                try:
                    feats = fut.result()
                except Exception as e:
                    print(f"Error in thread for {path}: {e}")
                    skipped += 1
                    continue
                if feats is None:
                    skipped += 1
                    continue
                features_list.append(feats)
                self.image_paths.append(os.path.abspath(path))
                inserted += 1

        if features_list:
            features_array = np.asarray(features_list, dtype='float32')
            self.index.add(features_array)

        logging.info(f"Index build complete. Inserted: {inserted}, Skipped: {skipped}")
        return len(self.image_paths)

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
        feats = self.extract_features(image_path)
        if feats is None:
            print(f"Insert skipped (invalid image): {image_path}")
            return len(self.image_paths)
        self.index.add(np.array([feats]).astype('float32'))
        self.image_paths.append(os.path.abspath(image_path))
        return len(self.image_paths)

    def rebuild(self, folder_path):
        return self.create(folder_path)

    def search(self, query_image_path, k=5, threshold=0.8):
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self.extract_features(query_image_path)
        if q is None:
            return []
        distances, indices = self.index.search(np.array([q]).astype('float32'), k)
        if distances is None or len(distances) == 0:
            return []
        max_sim = float(np.max(distances[0])) if len(distances[0]) else 0.0
        if not np.isfinite(max_sim) or max_sim <= 1e-8:
            return []
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            sim = float(dist) / max_sim
            if sim > threshold:
                results.append((self.image_paths[idx], float(sim)))
        return results

# =============================
# Indexer init/persist
# =============================
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

# =============================
# Flask endpoints
# =============================
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    logging.error(traceback.format_exc())
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.route('/open_url')
def open_website():
    try:
        port = current_settings.get('port_number', 5002)
        url = f"http://127.0.0.1:{port}"
        webbrowser.open_new_tab(url)
    except Exception as e:
        return jsonify({"error": f"Failed to open URL: {str(e)}"}), 500

@app.route('/')
def read_root():
    return render_template('index.html')

@app.route('/setup')
def setup():
    current = {
        'port_number': config['Settings'].get('port_number', ''),
        'previous_port_numbers': config['Settings'].get('previous_port_numbers', '').split(',')
    }
    return render_template('settings.html', current_settings=current)

# settings helpers

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'port_number': PORT, 'network_path': ''}
    except json.JSONDecodeError as e:
        print(f"Error loading settings: {e}")
        return {'port_number': PORT, 'network_path': ''}


def save_settings(port_number, network_path):
    settings = {'port_number': port_number, 'network_path': network_path}
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

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
        setup_file = 'setup.json'
        if os.path.exists(setup_file):
            with open(setup_file, 'r') as f:
                settings = json.load(f)
            network_path = settings.get('network_path', None)
        else:
            network_path = None

        folder_path = request.json.get('folder_path', network_path)
        max_workers = request.json.get('max_workers')  # optional

        if folder_path is None:
            return jsonify({"error": "No folder path provided and network_path is not set"}), 400
        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder path does not exist"}), 400

        CURRENT_FOLDER_PATH = folder_path
        indexer = FastImageIndexer(folder_path, max_workers=max_workers)
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
            image = request.files['image']
            if image.filename == '':
                return jsonify({"error": "No selected file"}), 400
            new_path = os.path.join(CURRENT_FOLDER_PATH, image.filename)
            image.save(new_path)
            num_images = indexer.insert(new_path)
            save_indexer_state()
            return jsonify({"message": f"Image inserted. Total images: {num_images}"})
        elif 'folder_path' in request.form:
            folder_path = request.form['folder_path']
            if not os.path.exists(folder_path):
                return jsonify({"error": "Folder path does not exist"}), 400
            inserted_count = 0
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif')):
                        image_path = os.path.join(root, filename)
                        before = len(indexer.image_paths)
                        indexer.insert(image_path)
                        after = len(indexer.image_paths)
                        if after > before:
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

        valid_exts = ('.jpg', '.jpeg', '.png', '.jfif')
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(valid_exts):
                    image_files.append(os.path.join(root, filename))

        inserted_count, skipped_count = 0, 0
        with ThreadPoolExecutor(max_workers=indexer.max_workers) as ex:
            futures = {ex.submit(indexer.insert, p): p for p in image_files}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                before = fut  # just to avoid linter warnings
                # insert() already counts only when success
            # recount after
        total_images = len(indexer.image_paths)
        # We cannot easily know how many were skipped without extra return codes; report total.
        save_indexer_state()
        result = {
            "message": f"Bulk insert complete. Total images in index: {total_images}",
            "total_images": total_images
        }
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
            full_path = next((p for p in indexer.image_paths if p.endswith(filename)), None)
            if full_path is None:
                errors.append(f"File {filename} not found in the index")
                continue
            if not os.path.exists(full_path):
                errors.append(f"File {filename} not found on disk")
                continue
            os.remove(full_path)
            idx = indexer.image_paths.index(full_path)
            indexer.image_paths.pop(idx)
            try:
                indexer.index.remove_ids(np.array([idx]))
            except Exception:
                pass
            successes.append(f"File {filename} successfully deleted")
        save_indexer_state()
        return jsonify({"messages": successes, "errors": errors})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# @app.route('/search', methods=['POST'])
# def search():
#     global indexer
#     try:
#         if indexer is None:
#             return jsonify({"error": "Model not trained yet. Use /train first."}), 400
#         if request.content_type.startswith('application/json'):
#             data = request.json
#         elif request.content_type.startswith('multipart/form-data'):
#             data = request.form
#         else:
#             return jsonify({"error": f"Unsupported content type: {request.content_type}"}), 400

#         if 'image' in request.files:
#             image = request.files['image']
#             if image.filename == '':
#                 return jsonify({"error": "No selected file"}), 400
#             image_data = image.read()
#             original_filename = image.filename
#         elif 'image' in data:
#             try:
#                 image_data = base64.b64decode(data['image'].split(',')[1])
#                 original_filename = "captured_image.jpg"
#             except:
#                 return jsonify({"error": "Invalid base64 image data"}), 400
#         else:
#             return jsonify({"error": "No image file or data provided"}), 400

#         temp_dir = tempfile.gettempdir()
#         unique_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#         temp_image_path = os.path.join(temp_dir, f'search_{unique_id}.jpg')
#         try:
#             with open(temp_image_path, 'wb') as f:
#                 f.write(image_data)
#         except Exception as e:
#             return jsonify({"error": f"Failed to save search image: {str(e)}"}), 500

#         Number_Of_Images_Req = int(data.get('Number_Of_Images_Req', 5))
#         Similarity_Percentage_str = data.get('Similarity_Percentage')
#         if Similarity_Percentage_str and str(Similarity_Percentage_str).strip():
#             Similarity_Percentage = float(Similarity_Percentage_str) / 100
#             if Similarity_Percentage == 1.0:
#                 Similarity_Percentage = 0.9999
#         else:
#             Similarity_Percentage = 0.8

#         results = indexer.search(temp_image_path, k=Number_Of_Images_Req, threshold=Similarity_Percentage)
#         try:
#             os.remove(temp_image_path)
#         except Exception:
#             pass

#         input_image_base64 = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"
#         input_result = {
#             "image_name": f"Input: {original_filename}",
#             "url": input_image_base64,
#             "similarity": 1.0,
#             "is_input": True
#         }

#         similar_results = []
#         for filename, sim in results:
#             image_name = os.path.basename(filename)
#             similar_results.append({
#                 "image_name": image_name,
#                 "url": url_for('serve_image', filename=image_name, _external=True),
#                 "similarity": float(sim),
#                 "is_input": False
#             })

#         max_similar = max(0, Number_Of_Images_Req - 1)
#         final_results = [input_result] + similar_results[:max_similar]

#         return jsonify({
#             "results": final_results,
#             "Status_Code": 200,
#             "Response": "Success!"
#         })
#     except Exception as e:
#         app.logger.error(traceback.format_exc())
#         return jsonify({"error": str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    Multi-image search API.
    Accepts:
      - multipart/form-data with one or more files under key 'images' (preferred)
      - multipart/form-data with a single file under key 'image' (backward compatible)
      - application/json with {'image': 'data:image/...;base64,....'} (legacy camera/base64)

    Body params (same as before):
      - Number_Of_Images_Req (int): how many total items to return PER INPUT including the input
      - Similarity_Percentage (0..100): threshold (exclusive of 100% -> coerced to 99.99%)
    """
    global indexer
    try:
        # ---- safety: model must be trained
        if indexer is None or indexer.index is None or indexer.index.ntotal == 0:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400

        # ---- extract body & files
        content_type = request.content_type or ""
        if content_type.startswith('application/json'):
            data = request.get_json(silent=True) or {}
            # JSON mode supports only a single base64 image (legacy)
            base64_image = data.get('image')
            files = []
            if not base64_image:
                return jsonify({"error": "No image data provided"}), 400
        elif content_type.startswith('multipart/form-data'):
            data = request.form
            # Preferred: multiple files under 'images'
            files = request.files.getlist('images')
            # Back-compat: single file under 'image'
            if not files and 'image' in request.files:
                single = request.files['image']
                if single and single.filename:
                    files = [single]
            base64_image = None
            if not files and 'image' in data:
                # edge case: base64 passed inside form field named 'image'
                base64_image = data.get('image')
        else:
            return jsonify({"error": f"Unsupported content type: {content_type}"}), 400

        # ---- parse params
        try:
            k_req = int((data or {}).get('Number_Of_Images_Req', 5))
        except Exception:
            k_req = 5
        k_req = max(1, k_req)  # at least show the input

        sim_str = (data or {}).get('Similarity_Percentage', None)
        if sim_str is None and content_type.startswith('application/json'):
            sim_str = (request.get_json(silent=True) or {}).get('Similarity_Percentage', None)
        if sim_str is not None and str(sim_str).strip() != "":
            try:
                sim_thr = float(sim_str) / 100.0
                # exactly 100 should not filter everything out; cap slightly below 1.0
                if sim_thr >= 1.0:
                    sim_thr = 0.9999
            except Exception:
                sim_thr = 0.8
        else:
            sim_thr = 0.8

        # ---- helper to process one in-memory image (PIL bytes) through the indexer
        def _search_one_tmp(jpeg_bytes: bytes, original_name: str):
            temp_dir = tempfile.gettempdir()
            unique_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            temp_image_path = os.path.join(temp_dir, f'search_{unique_id}.jpg')
            with open(temp_image_path, 'wb') as f:
                f.write(jpeg_bytes)

            # run ANN search
            results = indexer.search(temp_image_path, k=k_req, threshold=sim_thr)

            # build payload: include the input preview + top (k_req-1) similar
            input_b64 = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('utf-8')}"
            input_result = {
                "image_name": f"Input: {original_name}",
                "url": input_b64,
                "similarity": 1.0,
                "is_input": True
            }

            similar_results = []
            for filename, sim in results:
                image_name = os.path.basename(filename)
                # serve existing disk file via /uploads/<name>
                similar_results.append({
                    "image_name": image_name,
                    "url": url_for('serve_image', filename=image_name, _external=True),
                    "similarity": float(sim),
                    "is_input": False
                })

            max_similar = max(0, k_req - 1)
            final_results = [input_result] + similar_results[:max_similar]

            # best-effort cleanup
            try:
                os.remove(temp_image_path)
            except Exception:
                pass

            return final_results

        multi_payload = []  # list of {input_name, results:[...]}

        # ---- case A: multiple (or single) files
        if files:
            for f in files:
                if not f or not f.filename:
                    continue
                jpeg_bytes = f.read()
                per_input_results = _search_one_tmp(jpeg_bytes, f.filename)
                multi_payload.append({
                    "input_image": f.filename,
                    "results": per_input_results
                })

        # ---- case B: base64 (JSON or form field)
        elif base64_image:
            try:
                # allow prefixed "data:image/jpeg;base64,...."
                if ',' in base64_image:
                    _, encoded = base64_image.split(',', 1)
                else:
                    encoded = base64_image
                jpeg_bytes = base64.b64decode(encoded)
            except Exception:
                return jsonify({"error": "Invalid base64 image data"}), 400

            per_input_results = _search_one_tmp(jpeg_bytes, "captured_image.jpg")
            multi_payload.append({
                "input_image": "captured_image.jpg",
                "results": per_input_results
            })

        # nothing processed?
        if not multi_payload:
            return jsonify({"error": "No valid image(s) provided"}), 400

        # response mirrors your style, but now groups per input image
        return jsonify({
            "multi_results": multi_payload,
            "Status_Code": 200,
            "Response": "Success!"
        })

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<path:filename>')
def serve_image(filename):
    try:
        full_path = next((path for path in indexer.image_paths if os.path.basename(path) == filename), None)
        if full_path is None or not os.path.exists(full_path):
            app.logger.error(f"File not found: {filename}")
            return jsonify({"error": f"Image {filename} not found"}), 404
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

if __name__ == '__main__':
    port_number = load_settings().get('port_number', 5002)
    if not os.path.exists('/.dockerenv') and not os.getenv('RUNNING_IN_DOCKER'):
        try:
            webbrowser.open_new_tab(f"http://127.0.0.1:{port_number}")
        except Exception as e:
            logging.warning(f"Skipping auto-open browser: {e}")
    logging.info(f"Starting server on port {port_number}")
    app.run(debug=False, port=int(port_number), host='0.0.0.0')
    
