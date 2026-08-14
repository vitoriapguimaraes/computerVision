import streamlit as st
import tkinter as tk
from tkinter import filedialog


def select_folder_callback(key):
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    selected_folder = filedialog.askdirectory()
    root.destroy()

    if selected_folder:
        st.session_state[key] = selected_folder


def render_folder_selector(label, default_path, key):
    """
    Renderiza um seletor de pastas com input de texto e botão nativo.
    Retorna o caminho selecionado.
    """
    if key not in st.session_state:
        st.session_state[key] = default_path

    col1, col2 = st.columns([4, 1])

    with col1:
        path_input = st.text_input(label, key=key)

    with col2:
        st.write("")  # Spacer
        st.write("")
        st.button(
            "📂 Selecionar",
            key=f"{key}_btn",
            on_click=select_folder_callback,
            args=(key,),
        )

    if path_input:
        return path_input.strip().strip('"').strip("'")
    return path_input


def render_file_uploader(
    label, type, accept_multiple_files=False, key_prefix="uploader", help=None
):
    """
    Renderiza um file_uploader com botão de limpar integrado.
    Retorna o(s) arquivo(s) carregado(s).
    """
    # Chave para controlar o reset do uploader
    session_key = f"{key_prefix}_counter"

    if session_key not in st.session_state:
        st.session_state[session_key] = 0

    def reset_uploader():
        st.session_state[session_key] += 1

    # Componente uploader
    uploaded_files = st.file_uploader(
        label,
        type=type,
        accept_multiple_files=accept_multiple_files,
        key=f"{key_prefix}_{st.session_state[session_key]}",
        help=help,
    )

    # Botão de reset (se houver arquivos)
    if uploaded_files:
        st.button(
            "🧹 Limpar arquivos", key=f"{key_prefix}_clean_btn", on_click=reset_uploader
        )

    return uploaded_files


def apply_global_style():
    """
    Injeta o CSS de design premium global para todo o app no início da execução da página.
    """
    st.markdown("""
    <style>
        /* Importar fonte moderna Inter do Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Aplicar fonte a todo o app */
        html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* Estilo de fundo do app escuro e suave */
        .stApp {
            background-color: #0b0c10 !important;
            color: #c5c6c7 !important;
        }
        
        /* Customização da Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0d0e12 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Cards Premium com Efeito de Vidro (Glassmorphism) */
        .custom-card, div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 12px !important;
            padding: 18px !important;
            margin-bottom: 15px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
            transition: all 0.25s ease-in-out !important;
        }
        .custom-card:hover {
            transform: translateY(-2px);
            border-color: rgba(26, 115, 232, 0.3) !important;
            box-shadow: 0 8px 30px rgba(26, 115, 232, 0.08) !important;
        }
        
        /* Botões customizados */
        div.stButton > button {
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 6px 18px !important;
            font-weight: 500 !important;
            background-color: rgba(255, 255, 255, 0.03) !important;
            transition: all 0.2s ease-in-out !important;
        }
        div.stButton > button:hover {
            background-color: rgba(26, 115, 232, 0.1) !important;
            border-color: #1a73e8 !important;
            color: #1a73e8 !important;
            box-shadow: 0 0 12px rgba(26, 115, 232, 0.2) !important;
        }
        
        /* Botões primários */
        div.stButton > button[kind="primary"] {
            background-color: #1a73e8 !important;
            color: white !important;
            border: none !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #1557b0 !important;
            box-shadow: 0 0 15px rgba(26, 115, 232, 0.4) !important;
        }
        
        /* Badges de Status reutilizáveis */
        .status-badge {
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-mapped {
            background-color: rgba(46, 125, 50, 0.15) !important;
            color: #81c784 !important;
            border: 1px solid rgba(76, 175, 80, 0.3) !important;
        }
        .status-pending {
            background-color: rgba(239, 108, 0, 0.15) !important;
            color: #ffb74d !important;
            border: 1px solid rgba(255, 152, 0, 0.3) !important;
        }
        
        /* Inputs e Text Areas */
        div[data-baseweb="input"], div[data-baseweb="textarea"] {
            background-color: rgba(255, 255, 255, 0.01) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
        }
        
        /* Ajuste fino da área principal */
        .block-container {
            max-width: 95% !important;
            padding-top: 1.5rem !important;
            padding-bottom: 1.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)


def render_footer():
    """
    Renderiza o footer minimalista na página principal e na sidebar.
    """
    footer_html = """
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;">
        Desenvolvido por <a href="https://github.com/vitoriapguimaraes" target="_blank" style="color: #888; text-decoration: none; font-weight: 500;">github.com/vitoriapguimaraes</a>
    </div>
    """
    st.sidebar.markdown(footer_html, unsafe_allow_html=True)
