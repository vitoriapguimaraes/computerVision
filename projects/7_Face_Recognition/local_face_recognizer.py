import os
import re
import json
import urllib.request
import cv2
import numpy as np

# URLs dos modelos ONNX oficiais do OpenCV Zoo
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface.onnx")

def download_onnx_models():
    """Garante que os arquivos de modelo YuNet e SFace ONNX sejam baixados."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        
    for name, url, path in [("YuNet (Detecção)", YUNET_URL, YUNET_PATH), 
                            ("SFace (Reconhecimento)", SFACE_URL, SFACE_PATH)]:
        if not os.path.exists(path):
            print(f"Baixando modelo {name} de {url}...")
            try:
                # User-Agent para evitar bloqueio em alguns servidores
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"Modelo {name} baixado com sucesso.")
            except Exception as e:
                raise Exception(f"Falha ao baixar {name}: {e}")

def get_yunet_detector(width, height):
    """Instancia o detector de face YuNet para as dimensões fornecidas."""
    download_onnx_models()
    try:
        # scoreThreshold redefinido para 0.5 para aumentar a sensibilidade
        detector = cv2.FaceDetectorYN.create(
            YUNET_PATH,
            "",
            (width, height),
            0.5,
            0.3,
            50
        )
        return detector
    except Exception as e:
        raise Exception(f"Erro ao instanciar o detector YuNet: {e}")

def get_sface_recognizer():
    """Instancia o reconhecedor SFace."""
    download_onnx_models()
    try:
        recognizer = cv2.FaceRecognizerSF.create(
            SFACE_PATH,
            ""
        )
        return recognizer
    except Exception as e:
        raise Exception(f"Erro ao instanciar o reconhecedor SFace: {e}")

def clean_volunteer_name(filename):
    """Extrai e limpa o nome do voluntário a partir do nome do arquivo."""
    base = os.path.splitext(filename)[0]
    # Remove números no final (ex: "Roseli Roque Ribeiro 1" -> "Roseli Roque Ribeiro")
    clean = re.sub(r'\s+\d+$', '', base).strip()
    return clean

def load_and_resize_image(img_path, max_width=1000):
    """
    Carrega uma imagem suportando caminhos com acentos (Unicode) no Windows
    e redimensiona mantendo o aspect ratio se a largura ultrapassar max_width.
    """
    try:
        if not os.path.exists(img_path):
            return None
        # Leitura binária segura para caminhos unicode no Windows
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
        print(f"Erro ao ler imagem {img_path}: {e}")
        return None

def build_people_embeddings_db(people_dir):
    """
    Varre a pasta people_dir, detecta rostos nas fotos de referência
    e extrai os embeddings criando/atualizando o banco de dados local.
    Salva em .embeddings.json.
    """
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
        name = clean_volunteer_name(filename)
        
        img = load_and_resize_image(img_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        try:
            detector = get_yunet_detector(w, h)
            _, faces = detector.detect(img)
            
            if faces is not None and len(faces) > 0:
                # Pegar a face com maior confiança
                best_face = faces[0]
                aligned_face = recognizer.alignCrop(img, best_face)
                feature = recognizer.feature(aligned_face)
                
                # Armazenar no dict convertido para lista
                embedding_list = feature.flatten().tolist()
                
                if name not in db:
                    db[name] = []
                db[name].append(embedding_list)
                print(f"Embedding extraído para {name} a partir de {filename}")
            else:
                print(f"Nenhum rosto detectado em {filename}")
        except Exception as e:
            print(f"Erro ao processar embedding de {filename}: {e}")
            
    # Salvar em arquivo JSON oculto
    try:
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Erro ao salvar arquivo de embeddings: {e}")
        
    return db

def load_people_embeddings_db(people_dir):
    """Carrega o banco de dados de embeddings locais."""
    db_path = os.path.join(people_dir, ".embeddings.json")
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return build_people_embeddings_db(people_dir)

def match_face_local(face_feature, db, threshold=0.363):
    """
    Compara o embedding extraído contra o banco de dados de voluntários
    usando similaridade de cosseno do SFace.
    """
    best_name = "Desconhecido"
    best_score = -1.0
    
    recognizer = get_sface_recognizer()
    
    for name, embeddings in db.items():
        for ref_emb_list in embeddings:
            ref_feature = np.array(ref_emb_list, dtype=np.float32).reshape(1, -1)
            
            # Calcular similaridade de cosseno
            score = recognizer.match(face_feature, ref_feature, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_name = name
                
    if best_score >= threshold:
        return best_name, float(best_score)
    return "Desconhecido", float(best_score)

def detect_and_recognize_faces_local(image_bytes, db):
    """
    Detecta e reconhece rostos na imagem fornecida usando o banco de embeddings local.
    Retorna lista de dicionários: [{"bbox": (x, y, w, h), "name": str, "score": float}]
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None or not db:
        return []
        
    h_orig, w_orig, _ = img.shape
    
    # Redimensionar para melhorar performance e taxa de detecção
    scale = 1.0
    if w_orig > 1000:
        scale = 1000.0 / w_orig
        img = cv2.resize(img, (1000, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        
    h, w, _ = img.shape
    try:
        detector = get_yunet_detector(w, h)
        _, faces = detector.detect(img)
    except Exception as e:
        print(f"Falha na detecção YuNet: {e}")
        return []
        
    results = []
    if faces is not None:
        try:
            recognizer = get_sface_recognizer()
            for face_info in faces:
                # Coordenadas do box detectado no tamanho redimensionado
                x, y, w_box, h_box = face_info[0:4]
                
                # Mapear de volta para a resolução original
                bbox = (
                    int(x / scale),
                    int(y / scale),
                    int(w_box / scale),
                    int(h_box / scale)
                )
                
                # Alinhar e extrair características da imagem redimensionada
                aligned_face = recognizer.alignCrop(img, face_info)
                feature = recognizer.feature(aligned_face)
                
                # Comparar com voluntários conhecidos
                name, score = match_face_local(feature, db)
                
                results.append({
                    "bbox": bbox,
                    "name": name,
                    "score": score
                })
        except Exception as e:
            print(f"Erro durante reconhecimento de faces local: {e}")
            
    return results
