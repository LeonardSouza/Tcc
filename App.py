import streamlit as st
import os
import shutil
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from transformers import pipeline
from utils.model import classify_images

# Configurações iniciais
os.environ["PYTORCH_JIT"] = "0"
st.set_page_config(page_title="Classificador de Imagens")
device = 0 if torch.cuda.is_available() else -1

# Sessão
if "classificado" not in st.session_state:
    st.session_state["classificado"] = False
if "Imagens" not in st.session_state:
    st.session_state["Imagens"] = None
if "CaminhoImagens" not in st.session_state:
    st.session_state["CaminhoImagens"] = []
if "ImagensComErro" not in st.session_state:
    st.session_state["ImagensComErro"] = []
if "Etiqueta_nao_excluir" not in st.session_state:
    st.session_state["Etiqueta_nao_excluir"] = []
if "Etiqueta_excluir" not in st.session_state:
    st.session_state["Etiqueta_excluir"] = []
if "page" not in st.session_state:
    st.session_state.page = 'upload'

# Função para chamar modelo
def call_model():
    replace_directory("Resultados")

    if st.session_state["CaminhoImagens"]:
        labels = st.session_state["Etiqueta_nao_excluir"] + st.session_state["Etiqueta_excluir"]
        classify_images(
            st.session_state["CaminhoImagens"],
            destination="Resultados/",
            model_name="google/siglip-base-patch16-512",
            labels=labels,
            operation="copy",
            output_file="Resultados.csv",
            batch_size=50,
            verbose=False
        )
        st.session_state["classificado"] = True

    set_results()

def set_results():
    st.session_state.page = 'results'

def set_upload():
    st.session_state.page = 'upload'

def replace_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)  
    os.mkdir(path)  

def show_upload_page():
    st.title("Imagens para Classificação")

    st.markdown("""<h3 style='margin-bottom:0'>Etiquetas   
                     <span style='font-size: 0.6em;color: gray'>(separadas por vírgula)
                     </span>
                    </h3> """, unsafe_allow_html=True)
    st.markdown("""<label style='font-weight:600; font-size:1.1em; margin-bottom:0' >Não Excluir 
                     <span style='font-size:0.9em; color:gray'>(pessoa, animal, comida...)
                     </span>
                    </label>""", unsafe_allow_html=True)
    st.session_state["Etiqueta_nao_excluir"] = st.text_input(label="", value="", key="nao_excluir", label_visibility="collapsed").split(",")

    st.markdown("""<label style='font-weight:600; font-size:1.1em'>Excluir 
                     <span style='font-size:0.9em; color:gray'>(print de tela, imagem borrada...)
                     </span>
                    </label>""", unsafe_allow_html=True)
    st.session_state["Etiqueta_excluir"] = st.text_input(label="", value="", key="excluir", label_visibility="collapsed").split(",")

    st.divider()
    st.session_state["Imagens"] = st.file_uploader("Selecionar Imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Limpa listas para novo upload
    st.session_state["CaminhoImagens"] = []
    st.session_state["ImagensComErro"] = []

    if st.session_state["Imagens"]:
        replace_directory("ArquivosTemp")
        detector = pipeline(model="google/siglip-base-patch16-512", task="zero-shot-image-classification", device=device)

        for uploaded_file in st.session_state["Imagens"]:
            try:
                # Salva imagem temporária
                file_path = Path("ArquivosTemp") / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Verifica se é uma imagem válida
                with Image.open(file_path) as img:
                    img.verify()

                # Verifica se o modelo aceita a imagem
                detector(str(file_path), candidate_labels=["teste"])
                st.session_state["CaminhoImagens"].append(str(file_path))

            except Exception as e:
                st.session_state["ImagensComErro"].append({"arquivo": uploaded_file.name, "erro": str(e)})

        # Exibe feedback
        with st.expander("Validação de Imagens"):
            st.subheader("✅ Imagens Válidas")
            colunas_excluir = st.columns(5)
            index_col = 0   
            if st.session_state["CaminhoImagens"]:
                for caminho in st.session_state["CaminhoImagens"]:
                    with colunas_excluir[index_col]:
                        st.image(caminho, width=100, caption=os.path.basename(caminho))
                    index_col = (index_col + 1) % 5
            else:
                st.warning("Nenhuma imagem válida encontrada.")

            st.subheader("❌ Imagens com Erro")
            for item in st.session_state["ImagensComErro"]:
                st.error(f"{item['arquivo']}: {item['erro']}")

        st.button("Classificar", on_click=call_model)

def verify_category(valor):
    if valor in st.session_state["Etiqueta_nao_excluir"]:
        return "Não Excluir"
    elif valor in st.session_state["Etiqueta_excluir"]:
        return "Excluir"
    else:
        return ""

def show_results_page():
    st.title("Resultados da Classificação")
    st.button("⬅️ Voltar", on_click=set_upload)

    if st.session_state["classificado"]:
        st.success("Imagens classificadas com sucesso!")
        df = pd.read_csv("Resultados/Resultados.csv")
        df["category"] = df["label"].apply(verify_category)   

        st.subheader("✅ Imagens para não excluir.")
        st.write("Etiquetas: ", str(st.session_state["Etiqueta_nao_excluir"]).replace("[", "")
                                                                            .replace("'", "")
                                                                            .replace("]",""))
        colunas_nao_excluir = st.columns(5)
        index_col = 0
        
        for caminho in st.session_state["CaminhoImagens"]:
            try:
                if df.loc[df["image_name"] == caminho, "category"].iloc[0] == "Não Excluir":
                    with colunas_nao_excluir[index_col]:
                        st.image(caminho, width=100, caption=os.path.basename(caminho))
                    index_col = (index_col + 1) % 5
            except Exception as e:
                st.warning(f"Erro ao processar imagem {caminho}: {e}")

        st.subheader("❌ Imagens para excluir.")
        st.write("Etiquetas: ", str(st.session_state["Etiqueta_excluir"]).replace("[", "")
                                                                        .replace("'", "")
                                                                        .replace("]",""))

        colunas_excluir = st.columns(5)
        index_col = 0

        for caminho in st.session_state["CaminhoImagens"]:
            try:
                if df.loc[df["image_name"] == caminho, "category"].iloc[0] == "Excluir":
                    with colunas_excluir[index_col]:
                        st.image(caminho, width=100, caption=os.path.basename(caminho))
                    index_col = (index_col + 1) % 5
            except Exception as e:
                st.warning(f"Erro ao processar imagem {caminho}: {e}")


# Página atual
if st.session_state.page == 'upload':
    show_upload_page()
elif st.session_state.page == 'results':
    show_results_page()

