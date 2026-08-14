import streamlit as st
import os
import io
import json
import base64
from PIL import Image
import pandas as pd

# Importar utilitários
from ui import render_footer, render_folder_selector, apply_global_style
from accessibility_tools import (
    read_accessibility_table,
    write_accessibility_table,
    load_people_catalog,
    save_people_catalog,
    crop_face,
    describe_image_with_vision,
    describe_image_with_gemini
)
from local_face_recognizer import (
    load_people_embeddings_db,
    build_people_embeddings_db,
    detect_and_recognize_faces_local
)

# Configurações iniciais da página
st.set_page_config(
    page_title="Acessibilidade e Classificação de Fotos",
    page_icon="♿",
    layout="wide"
)
apply_global_style()


st.title("♿ Acessibilidade & Classificação de Fotos (GPCC)")
st.markdown(
    "Mapeie fotos locais e gere descrições alt-text de acessibilidade para o site e livro da associação, identificando voluntários de forma automatizada com IA ou localmente de forma gratuita."
)

# Inicializar estados da sessão
if "mapped_data" not in st.session_state:
    st.session_state.mapped_data = {}
if "people_catalog" not in st.session_state:
    st.session_state.people_catalog = {}
if "selected_img_name" not in st.session_state:
    st.session_state.selected_img_name = None
if "image_files_list" not in st.session_state:
    st.session_state.image_files_list = []
if "temp_edits" not in st.session_state:
    st.session_state.temp_edits = {}
if "face_cache" not in st.session_state:
    st.session_state.face_cache = {}
if "local_embeddings_db" not in st.session_state:
    st.session_state.local_embeddings_db = {}

# --- Configurações na Sidebar ---
st.sidebar.subheader("⚙️ Configurações de Origem")

# Escolha do provedor de IA
ai_provider = st.sidebar.selectbox(
    "Provedor de Vision AI:",
    ["Google Gemini (Grátis)", "OpenAI GPT-4o-mini"]
)

# Chave API Dinâmica
if ai_provider == "Google Gemini (Grátis)":
    default_gemini = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    api_key_input = st.sidebar.text_input(
        "Gemini API Key:",
        value=default_gemini,
        type="password",
        help="Obtenha uma chave grátis no Google AI Studio para análise gratuita."
    )
else:
    default_openai = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key (Vision):",
        value=default_openai,
        type="password"
    )

# Pastas e Arquivos padrão
default_photos_dir = r"C:\Users\Vitoria\Documents\GitHub\REPO_ON_WORKING\soscancer_latex\photos"
default_txt_path = os.path.join(default_photos_dir, "acessibilidade_fotos.txt")
default_catalog_path = os.path.join(default_photos_dir, ".people_catalog.json")

# Seletor de Pastas
selected_photos_dir = render_folder_selector("Pasta das Fotos:", default_photos_dir, "photos_dir_key")

txt_file_path = st.sidebar.text_input("Arquivo de Mapeamento (.txt):", value=default_txt_path)
catalog_file_path = st.sidebar.text_input("Catálogo de Pessoas (.json):", value=default_catalog_path)

# Função auxiliar para recarregar
def load_all_project_data():
    if os.path.exists(selected_photos_dir):
        valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
        files = [
            f for f in os.listdir(selected_photos_dir)
            if f.lower().endswith(valid_extensions)
        ]
        st.session_state.image_files_list = sorted(files)
        st.session_state.mapped_data = read_accessibility_table(txt_file_path)
        st.session_state.people_catalog = load_people_catalog(catalog_file_path)
        
        # Carregar embeddings locais
        people_dir = os.path.join(selected_photos_dir, "people")
        st.session_state.local_embeddings_db = load_people_embeddings_db(people_dir)
        
        if files and not st.session_state.selected_img_name:
            st.session_state.selected_img_name = files[0]
        return len(files)
    return 0

# Botão para Carregar/Recarregar Diretório
if st.sidebar.button("Carregar Dados do Projeto 🔄", type="primary", use_container_width=True):
    count = load_all_project_data()
    if count > 0:
        st.sidebar.success(f"Diretório lido! {count} fotos encontradas.")
    else:
        st.sidebar.error("Pasta selecionada não existe ou não contém imagens.")

# Inicializar dados automaticamente na primeira execução se o diretório padrão existir
if not st.session_state.image_files_list and os.path.exists(selected_photos_dir):
    load_all_project_data()

# --- Navegação por Abas ---
tab_mapper, tab_catalog = st.tabs(["🖼️ Mapeador de Acessibilidade", "👥 Catálogo de Voluntários"])

# ==================== TAB 2: CATÁLOGO DE VOLUNTÁRIOS ====================
with tab_catalog:
    st.subheader("👥 Cadastro de Voluntários do GPCC")
    st.markdown(
        "Cadastre e gerencie a descrição física dos voluntários da associação. Isso ajuda a IA a reconhecê-los."
    )
    
    # Criar Novo Voluntário
    with st.expander("➕ Cadastrar Novo Voluntário", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            new_name = st.text_input("Nome do Voluntário:")
        with c2:
            new_desc = st.text_area(
                "Descrição Física Detalhada:",
                placeholder="Ex: Homem de cabelos brancos curtos, usa óculos de grau redondos, geralmente veste a camisa amarela do GPCC."
            )
            
        if st.button("Salvar Cadastro 💾"):
            if new_name.strip() and new_desc.strip():
                name_key = new_name.strip()
                if name_key not in st.session_state.people_catalog:
                    st.session_state.people_catalog[name_key] = {
                        "description": new_desc.strip(),
                        "faces": []
                    }
                else:
                    st.session_state.people_catalog[name_key]["description"] = new_desc.strip()
                    
                save_people_catalog(catalog_file_path, st.session_state.people_catalog)
                st.success(f"Cadastro de {name_key} realizado com sucesso!")
                st.rerun()
            else:
                st.warning("Preencha o nome e a descrição do voluntário.")
                
    # Listar Voluntários Cadastrados
    if st.session_state.people_catalog:
        st.write("---")
        st.subheader("Voluntários Catalogados")
        
        # Grid para exibir
        cols = st.columns(3)
        for i, (name, info) in enumerate(st.session_state.people_catalog.items()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"### **{name}**")
                    st.markdown(f"*{info.get('description', 'Sem descrição física.')}*")
                    
                    # Exibir rostos associados (se existirem)
                    faces = info.get("faces", [])
                    if faces:
                        st.markdown(f"**Rostos cadastrados ({len(faces)}):**")
                        face_cols = st.columns(min(len(faces), 4))
                        for f_idx, face_b64 in enumerate(faces[:4]):
                            with face_cols[f_idx]:
                                try:
                                    face_bytes = base64.b64decode(face_b64)
                                    st.image(face_bytes, use_container_width=True)
                                except Exception:
                                    pass
                                    
                    # Botão para deletar voluntário
                    if st.button(f"Remover {name} 🗑️", key=f"del_person_{name}_{i}"):
                        del st.session_state.people_catalog[name]
                        save_people_catalog(catalog_file_path, st.session_state.people_catalog)
                        st.success(f"{name} removido.")
                        st.rerun()
                        
        # Seção Banco de Rostos Local Offline
        st.write("---")
        st.subheader("⚙️ Banco de Rostos Local (Reconhecimento Offline)")
        st.caption("Ao adicionar novas fotos de referência à pasta `photos/people/`, recalcule as assinaturas faciais abaixo.")
        if st.button("Recalcular/Atualizar Banco de Rostos Local 👥", type="primary", use_container_width=True):
            people_dir = os.path.join(selected_photos_dir, "people")
            with st.spinner("Analisando fotos na pasta 'people' e gerando assinaturas locais (embeddings)..."):
                try:
                    db = build_people_embeddings_db(people_dir)
                    st.session_state.local_embeddings_db = db
                    st.success(f"Banco de rostos local atualizado com sucesso! {len(db)} voluntários catalogados.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Erro ao recalcular banco local: {ex}")
    else:
        st.info("Nenhum voluntário cadastrado no catálogo ainda.")

# ==================== TAB 1: MAPEADOR DE IMAGENS ====================
with tab_mapper:
    if not st.session_state.image_files_list:
        st.warning(
            "Nenhuma foto encontrada. Por favor, certifique-se de carregar os dados do projeto no menu lateral."
        )
    else:
        # Layout de duas colunas
        col_list, col_editor = st.columns([1, 2.5])
        
        # --- Lado Esquerdo: Lista de Imagens ---
        with col_list:
            st.subheader("Fila de Imagens")
            
            # Campo de busca rápido
            search_query = st.text_input("🔍 Filtrar fotos:", "").lower()
            
            # Filtro por Status
            status_filter = st.selectbox("Filtrar por Status:", ["Todas", "Pendentes", "Mapeadas"])
            
            for file_name in st.session_state.image_files_list:
                # Filtragem por busca
                if search_query and search_query not in file_name.lower():
                    continue
                    
                # Determinar status
                base_name = os.path.splitext(file_name)[0]
                table_key = base_name.upper()
                is_mapped = table_key in st.session_state.mapped_data
                
                # Filtragem por status
                if status_filter == "Pendentes" and is_mapped:
                    continue
                if status_filter == "Mapeadas" and not is_mapped:
                    continue
                
                badge_label = "Mapeado" if is_mapped else "Pendente"
                
                if st.button(
                    f"{file_name[:25]}... ({badge_label})" if len(file_name) > 25 else f"{file_name} ({badge_label})",
                    key=f"select_{file_name}",
                    use_container_width=True,
                    type="secondary" if st.session_state.selected_img_name != file_name else "primary"
                ):
                    st.session_state.selected_img_name = file_name
                    st.rerun()

        # --- Lado Direito: Editor de Acessibilidade ---
        with col_editor:
            selected_name = st.session_state.selected_img_name
            if selected_name:
                img_path = os.path.join(selected_photos_dir, selected_name)
                
                try:
                    with open(img_path, "rb") as img_file:
                        img_bytes = img_file.read()
                    pil_image = Image.open(io.BytesIO(img_bytes))
                except Exception as e:
                    st.error(f"Erro ao carregar imagem {selected_name}: {e}")
                    st.stop()
                
                st.subheader(f"🖼️ Editando: {selected_name}")
                
                # Layout das colunas de visualização e edição
                edit_left, edit_right = st.columns([1, 1.2])
                
                base_name = os.path.splitext(selected_name)[0]
                table_key = base_name.upper()
                
                # Pré-carregar reconhecimento local de rostos na inicialização da imagem para preencher voluntários
                if table_key not in st.session_state.temp_edits:
                    people_dir = os.path.join(selected_photos_dir, "people")
                    if not st.session_state.local_embeddings_db:
                        st.session_state.local_embeddings_db = load_people_embeddings_db(people_dir)
                    
                    # Rodar detecção e matching offline local
                    auto_matches = detect_and_recognize_faces_local(img_bytes, st.session_state.local_embeddings_db)
                    st.session_state.face_cache[selected_name] = auto_matches
                    
                    # Voluntários identificados localmente
                    auto_people = [m["name"] for m in auto_matches if m["name"] != "Desconhecido"]
                    
                    if table_key in st.session_state.mapped_data:
                        entry = st.session_state.mapped_data[table_key]
                        st.session_state.temp_edits[table_key] = {
                            "title": entry["title"],
                            "event": entry["event"],
                            "description": entry["description"],
                            "people": list(set(auto_people))
                        }
                    else:
                        st.session_state.temp_edits[table_key] = {
                            "title": base_name,
                            "event": "",
                            "description": "",
                            "people": list(set(auto_people))
                        }
                
                current_edit = st.session_state.temp_edits[table_key]
                
                with edit_left:
                    st.image(pil_image, caption=selected_name, use_container_width=True)
                    
                    # Automação por IA
                    st.write("")
                    st.markdown("### 🧠 Automação de Acessibilidade")
                    if st.button("Descrever com Vision AI ⚡", type="primary", use_container_width=True):
                        if not api_key_input:
                            st.error(f"Chave de API do {ai_provider} não fornecida.")
                        else:
                            with st.spinner(f"Analisando imagem com Vision AI ({ai_provider})..."):
                                try:
                                    if ai_provider == "Google Gemini (Grátis)":
                                        ai_results = describe_image_with_gemini(
                                            image_bytes=img_bytes,
                                            api_key=api_key_input
                                        )
                                    else:
                                        ai_results = describe_image_with_vision(
                                            image_bytes=img_bytes,
                                            api_key=api_key_input,
                                            known_people=st.session_state.people_catalog
                                        )
                                    
                                    # Atualizar os campos temporários
                                    st.session_state.temp_edits[table_key]["title"] = ai_results.get("suggested_title", base_name)
                                    st.session_state.temp_edits[table_key]["event"] = ai_results.get("event", "")
                                    st.session_state.temp_edits[table_key]["description"] = ai_results.get("description", "")
                                    
                                    # Se a IA sugeriu pessoas conhecidas, adicioná-las
                                    ai_people = ai_results.get("identified_people", [])
                                    for p in ai_people:
                                        if p not in st.session_state.temp_edits[table_key]["people"]:
                                            st.session_state.temp_edits[table_key]["people"].append(p)
                                            
                                    st.success("Descrição gerada com sucesso!")
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Erro na análise de visão: {err}")
                                    
                with edit_right:
                    st.markdown("### Dados da Imagem")
                    
                    # Campos de input
                    edited_title = st.text_input(
                        "Título do Documento (nome na tabela/LaTeX):",
                        value=current_edit["title"],
                        key=f"title_input_{selected_name}"
                    )
                    
                    edited_event = st.text_input(
                        "Evento / Categoria:",
                        value=current_edit["event"],
                        key=f"event_input_{selected_name}"
                    )
                    
                    edited_description = st.text_area(
                        "Descrição (Alt Text de Acessibilidade):",
                        value=current_edit["description"],
                        height=150,
                        key=f"desc_input_{selected_name}"
                    )
                    
                    # Atualizar valores na sessão conforme digita
                    st.session_state.temp_edits[table_key]["title"] = edited_title
                    st.session_state.temp_edits[table_key]["event"] = edited_event
                    st.session_state.temp_edits[table_key]["description"] = edited_description
                    
                    # Pessoas marcadas
                    st.markdown("#### Voluntários Marcados")
                    tagged_people = st.session_state.temp_edits[table_key].get("people", [])
                    
                    if tagged_people:
                        # Mostrar tags como badges
                        cols_badge = st.columns(len(tagged_people) + 1)
                        for p_idx, person_name in enumerate(tagged_people):
                            with cols_badge[p_idx]:
                                if st.button(f"{person_name} ❌", key=f"rm_p_{person_name}_{selected_name}_{p_idx}"):
                                    st.session_state.temp_edits[table_key]["people"].remove(person_name)
                                    st.rerun()
                    else:
                        st.caption("Nenhum voluntário marcado nesta imagem.")
                        
                    # Opção de marcar manualmente
                    if st.session_state.people_catalog:
                        manual_person = st.selectbox(
                            "Adicionar voluntário à foto:",
                            options=["Selecione um voluntário..."] + list(st.session_state.people_catalog.keys()),
                            key=f"manual_person_add_{selected_name}"
                        )
                        if manual_person != "Selecione um voluntário...":
                            if manual_person not in st.session_state.temp_edits[table_key]["people"]:
                                st.session_state.temp_edits[table_key]["people"].append(manual_person)
                            st.rerun()
                            
                # --- Seção de Detecção de Rostos e Assinatura de Faces (Google Fotos Local) ---
                st.divider()
                st.markdown("### 👤 Banco de Rostos Detectados nesta Imagem")
                st.caption(
                    "Os rostos abaixo foram localizados e comparados de forma offline e gratuita com a pasta 'people'."
                )
                
                # Garantir que a detecção/reconhecimento foi rodado
                if selected_name not in st.session_state.face_cache:
                    people_dir = os.path.join(selected_photos_dir, "people")
                    if not st.session_state.local_embeddings_db:
                        st.session_state.local_embeddings_db = load_people_embeddings_db(people_dir)
                    st.session_state.face_cache[selected_name] = detect_and_recognize_faces_local(img_bytes, st.session_state.local_embeddings_db)
                    
                matches = st.session_state.face_cache[selected_name]
                
                if matches:
                    st.info(f"Foram encontrados **{len(matches)}** rostos nesta imagem.")
                    face_cols = st.columns(min(len(matches), 4))
                    
                    for f_idx, match_info in enumerate(matches):
                        bbox = match_info["bbox"]
                        matched_name = match_info["name"]
                        similarity = match_info.get("score", 0.0)
                        
                        col_idx = f_idx % 4
                        with face_cols[col_idx]:
                            face_img_bytes = crop_face(img_bytes, bbox)
                            if face_img_bytes:
                                with st.container(border=True):
                                    st.image(face_img_bytes, caption=f"Rosto #{f_idx+1}", use_container_width=True)
                                    
                                    # Exibir sugestão local
                                    if matched_name != "Desconhecido":
                                        st.success(f"Sugerido: **{matched_name}** ({similarity:.2f})")
                                    else:
                                        st.caption("Não reconhecido.")
                                        
                                    # Confirmar ou redefinir manualmente
                                    p_options = ["Escolha...", "Desconhecido"] + list(st.session_state.people_catalog.keys())
                                    
                                    # Pré-selecionar o voluntário reconhecido localmente
                                    default_idx = 0
                                    if matched_name in st.session_state.people_catalog:
                                        default_idx = p_options.index(matched_name)
                                    elif matched_name == "Desconhecido":
                                        default_idx = 1
                                        
                                    assigned_person = st.selectbox(
                                        f"Quem é o Rosto #{f_idx+1}?",
                                        options=p_options,
                                        index=default_idx,
                                        key=f"assign_face_{selected_name}_{f_idx}"
                                    )
                                    
                                    if assigned_person not in ("Escolha...", "Desconhecido"):
                                        # Adicionar aos voluntários marcados na foto
                                        if assigned_person not in st.session_state.temp_edits[table_key]["people"]:
                                            st.session_state.temp_edits[table_key]["people"].append(assigned_person)
                                            
                                        # Adicionar rosto de referência ao catálogo
                                        b64_face = base64.b64encode(face_img_bytes).decode("utf-8")
                                        if b64_face not in st.session_state.people_catalog[assigned_person]["faces"]:
                                            if len(st.session_state.people_catalog[assigned_person]["faces"]) >= 5:
                                                st.session_state.people_catalog[assigned_person]["faces"].pop(0)
                                            st.session_state.people_catalog[assigned_person]["faces"].append(b64_face)
                                            
                                            save_people_catalog(catalog_file_path, st.session_state.people_catalog)
                                            st.success(f"Referência salva!")
                                            st.rerun()
                else:
                    st.caption("Nenhum rosto claro detectado nesta foto pelo algoritmo local.")
                
                # --- Salvamento e Sincronização ---
                st.divider()
                
                c_save1, c_save2 = st.columns(2)
                
                with c_save1:
                    if st.button("Confirmar Edição desta Foto 🔒", use_container_width=True, type="primary"):
                        final_desc = edited_description
                        people_list = st.session_state.temp_edits[table_key]["people"]
                        if people_list:
                            people_str = ", ".join(people_list)
                            if not any(person.lower() in final_desc.lower() for person in people_list):
                                final_desc += f" (Identificados na foto: {people_str})"
                        
                        st.session_state.mapped_data[table_key] = {
                            "title": edited_title,
                            "event": edited_event,
                            "description": final_desc
                        }
                        st.success(f"Alterações gravadas localmente!")
                        st.rerun()
                        
                with c_save2:
                    if st.button("Sincronizar e Gravar no Arquivo Geral 💾", use_container_width=True):
                        # Fazer backup
                        if os.path.exists(txt_file_path):
                            bak_path = txt_file_path + ".bak"
                            try:
                                with open(txt_file_path, "r", encoding="utf-8") as orig:
                                    with open(bak_path, "w", encoding="utf-8") as bak:
                                        bak.write(orig.read())
                            except Exception:
                                pass
                                
                        try:
                            write_accessibility_table(txt_file_path, st.session_state.mapped_data)
                            st.success(f"Arquivo '{os.path.basename(txt_file_path)}' atualizado com sucesso!")
                        except Exception as e:
                            st.error(f"Erro ao salvar arquivo: {e}")
                            
            else:
                st.info("Selecione uma imagem na barra lateral para começar a mapear a acessibilidade.")

render_footer()
