import os
import re
import json
import urllib.request
import cv2
import numpy as np

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface.onnx")

def download_onnx_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url, path in [("YuNet (Detecção)", YUNET_URL, YUNET_PATH), 
                            ("SFace (Reconhecimento)", SFACE_URL, SFACE_PATH)]:
        if not os.path.exists(path):
            print(f"Baixando modelo {name} de {url}...")
            try:
                req = urllib.request.Request(
                    url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                raise Exception(f"Falha ao baixar {name}: {e}")

def get_yunet_detector(width, height):
    download_onnx_models()
    try:
        return cv2.FaceDetectorYN.create(YUNET_PATH, "", (width, height), 0.5, 0.3, 50)
    except Exception as e:
        raise Exception(f"Erro ao instanciar o detector YuNet: {e}")

def get_sface_recognizer():
    download_onnx_models()
    try:
        return cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    except Exception as e:
        raise Exception(f"Erro ao instanciar o reconhecedor SFace: {e}")

def load_and_resize_image(img_path, max_width=1000):
    try:
        if not os.path.exists(img_path):
            return None
        np_arr = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w, _ = img.shape
        if w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        return None

def build_people_embeddings_db(people_dir):
    db_path = os.path.join(people_dir, ".embeddings.json")
    db = {}
    if not os.path.exists(people_dir):
        return db
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(people_dir) if f.lower().endswith(valid_extensions)]
    if not image_files:
        return db
    recognizer = get_sface_recognizer()
    for filename in image_files:
        img_path = os.path.join(people_dir, filename)
        name = os.path.splitext(filename)[0]
        name = re.sub(r'\s+\d+$', '', name).strip()
        img = load_and_resize_image(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        try:
            detector = get_yunet_detector(w, h)
            _, faces = detector.detect(img)
            if faces is not None and len(faces) > 0:
                best_face = faces[0]
                aligned_face = recognizer.alignCrop(img, best_face)
                feature = recognizer.feature(aligned_face)
                if name not in db:
                    db[name] = []
                db[name].append(feature.flatten().tolist())
        except Exception:
            continue
    try:
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=4, ensure_ascii=False)
    except Exception:
        pass
    return db

def load_people_embeddings_db(people_dir):
    db_path = os.path.join(people_dir, ".embeddings.json")
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return build_people_embeddings_db(people_dir)

def match_face_local(face_feature, db, recognizer, threshold=0.363):
    best_name = "Desconhecido"
    best_score = -1.0
    for name, embeddings in db.items():
        for ref_emb_list in embeddings:
            ref_feature = np.array(ref_emb_list, dtype=np.float32).reshape(1, -1)
            score = recognizer.match(face_feature, ref_feature, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_name = name
    if best_score >= threshold:
        return best_name, float(best_score)
    return "Desconhecido", float(best_score)
