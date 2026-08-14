import os
import re
import json
import base64
import cv2
import numpy as np
from PIL import Image
import io
from openai import OpenAI

def read_accessibility_table(file_path):
    """
    Lê o arquivo de acessibilidade e faz o parse da tabela markdown.
    Retorna um dicionário: {titulo_limpo: {"title": title, "event": event, "description": description}}
    """
    data_dict = {}
    if not os.path.exists(file_path):
        return data_dict

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            line_str = line.strip()
            # Ignora linhas vazias, comentários ou separadores da tabela
            if not line_str or line_str.startswith("#") or "---" in line_str:
                continue
            
            # Formato esperado: | Titulo do documento | Evento | Descrição |
            parts = [p.strip() for p in line_str.split("|")]
            # Filtra partes vazias no início e fim geradas pelo split em pipes nas bordas
            if len(parts) >= 4:
                # parts[0] é vazio (antes do primeiro |), parts[1] é Titulo, parts[2] é Evento, parts[3] é Descrição
                title = parts[1]
                event = parts[2]
                description = parts[3]
                
                # Se for o cabeçalho, ignora
                if title.lower() in ("titulo do documento", "título do documento", "titulo"):
                    continue
                
                # Chave única normalizada (caixa alta)
                key = title.strip().upper()
                data_dict[key] = {
                    "title": title,
                    "event": event,
                    "description": description
                }
    except Exception as e:
        print(f"Erro ao ler tabela de acessibilidade: {e}")
        
    return data_dict

def write_accessibility_table(file_path, data_dict):
    """
    Grava as informações de acessibilidade de volta no formato markdown original.
    Ordena as chaves alfabeticamente.
    """
    try:
        # Garante o diretório pai
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Acessibilidade das fotos\n\n")
            f.write("| Titulo do documento | Evento | Descrição |\n")
            f.write("| --- | --- | --- |\n")
            
            # Ordenar por título para manter organizado
            for key in sorted(data_dict.keys()):
                entry = data_dict[key]
                title = entry["title"].strip()
                event = entry["event"].strip()
                description = entry["description"].strip()
                
                # Escapar pipes caso existam no texto para não quebrar a tabela markdown
                title_esc = title.replace("|", "\\|")
                event_esc = event.replace("|", "\\|")
                desc_esc = description.replace("|", "\\|")
                
                f.write(f"| {title_esc} | {event_esc} | {desc_esc} |\n")
                
        return True
    except Exception as e:
        raise Exception(f"Erro ao salvar tabela: {str(e)}")

def load_people_catalog(catalog_path):
    """
    Carrega o arquivo JSON do catálogo de voluntários/pessoas.
    Estrutura: {
      "Name": {
         "description": "descrição física para ajudar a IA",
         "reference_faces": ["base64_img1", "base64_img2"]
      }
    }
    """
    if not os.path.exists(catalog_path):
        return {}
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_people_catalog(catalog_path, catalog):
    """
    Salva o catálogo de voluntários em arquivo JSON.
    """
    try:
        os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erro ao salvar catálogo de pessoas: {e}")
        return False

def detect_faces(image_bytes):
    """
    Detecta rostos usando o classificador Cascade do OpenCV.
    Retorna uma lista de tuplas (x, y, w, h)
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detecção com parâmetros que evitam falsos positivos mas pegam bem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [tuple(map(int, face)) for face in faces]

def crop_face(image_bytes, bbox, margin_ratio=0.2):
    """
    Recorta a face com uma margem de segurança.
    Retorna bytes da imagem recortada no formato PNG.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
        
    h_img, w_img, _ = img.shape
    x, y, w, h = bbox
    
    # Adicionar margem
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)
    
    face_crop = img[y1:y2, x1:x2]
    # Converter para RGB antes do PIL
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(face_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def describe_image_with_vision(image_bytes, api_key, known_people=None):
    """
    Usa a API Vision da OpenAI (gpt-4o-mini) para gerar os campos de acessibilidade.
    Também envia a lista de pessoas conhecidas para que a IA ajude a identificá-las se possível.
    """
    if not api_key:
        raise ValueError("Chave de API OpenAI ausente.")
        
    client = OpenAI(api_key=api_key)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    # Preparar a informação dos voluntários conhecidos para o prompt
    people_desc = ""
    if known_people:
        people_desc = "Aqui está uma lista de voluntários conhecidos e suas descrições físicas para ajudar a identificá-los:\n"
        for name, info in known_people.items():
            desc_str = info.get("description", "Sem descrição física")
            people_desc += f"- **{name}**: {desc_str}\n"
    
    prompt = f"""
    Você é um assistente especialista em acessibilidade para pessoas com deficiência visual e descrição de imagens históricas.
    Analise a imagem fornecida e responda no formato JSON.
    
    {people_desc}
    
    Siga estritamente as regras de preenchimento abaixo:
    1. **suggested_title**: Sugira um título curto, em caixa alta, representativo para a imagem (ex: "CAMPANHA DOACAO 4", "LEILAO 14"). Se a imagem já tiver um nome intuitivo sugerido, tente basear-se nele.
    2. **event**: Identifique ou infira qual é o evento ou categoria (ex: "Leilão de gado", "Campanha de doação", "Inauguração da Sede", "Confraternização final do ano"). Forneça a resposta em formato amigável como "Ano, Nome do Evento" se for dedutível, ou apenas "Nome do Evento".
    3. **description**: Escreva uma descrição detalhada de acessibilidade (Alt-Text) em português do Brasil. Descreva o que está acontecendo na cena, as pessoas presentes, roupas, cores, ambiente, sentimentos aparentes e elementos visuais importantes de forma fluida.
    4. **identified_people**: Retorne uma lista com o nome exato dos voluntários conhecidos que você conseguiu identificar com alta confiança na imagem, baseando-se na lista fornecida. Se não identificar ninguém com confiança, retorne uma lista vazia.
    
    Você deve responder APENAS com um objeto JSON válido, contendo exatamente as chaves:
    {{
      "suggested_title": "string",
      "event": "string",
      "description": "string",
      "identified_people": ["string"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=600,
            temperature=0.2
        )
        
        result_content = response.choices[0].message.content.strip()
        return json.loads(result_content)
        
    except Exception as e:
        raise Exception(f"Falha na chamada da API OpenAI Vision: {str(e)}")

def describe_image_with_gemini(image_bytes, api_key):
    """
    Usa a API Gemini 1.5 Flash para gerar os campos de acessibilidade.
    """
    if not api_key:
        raise ValueError("Chave de API Gemini ausente.")
        
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    image_data = {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }
    
    prompt = """
    Você é um assistente especialista em acessibilidade para pessoas com deficiência visual e descrição de imagens históricas.
    Analise a imagem fornecida e responda no formato JSON em português.
    
    Siga estritamente as regras de preenchimento abaixo:
    1. **suggested_title**: Sugira um título curto, em caixa alta, representativo para a imagem (ex: "CAMPANHA DOACAO 4", "LEILAO 14").
    2. **event**: Identifique ou infira qual é o evento ou categoria (ex: "Leilão de gado", "Campanha de doação", "Inauguração da Sede").
    3. **description**: Escreva uma descrição detalhada de acessibilidade (Alt-Text) em português do Brasil. Descreva o que está acontecendo na cena, as pessoas presentes, roupas, cores, ambiente, sentimentos aparentes e elementos visuais importantes de forma fluida.
    4. **identified_people**: Retorne uma lista vazia por padrão: []. (A identificação será feita localmente).
    
    Você deve responder APENAS com um objeto JSON válido, contendo exatamente as chaves:
    {
      "suggested_title": "string",
      "event": "string",
      "description": "string",
      "identified_people": []
    }
    """
    
    try:
        response = model.generate_content(
            [image_data, prompt],
            generation_config={"response_mime_type": "application/json"}
        )
        
        result_content = response.text.strip()
        return json.loads(result_content)
        
    except Exception as e:
        raise Exception(f"Falha na chamada da API Gemini: {str(e)}")

