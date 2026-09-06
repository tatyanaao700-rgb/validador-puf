import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 1. Configuração da página (DEVE SER O PRIMEIRO COMANDO)
st.set_page_config(page_title="Validador de Selo UV - PUF", page_icon="🔍", layout="centered")

# 2. Funções auxiliares (Recorte e Validação)
def recortar_centro(img_np):
    """Recorta os 60% centrais da imagem para focar no selo e eliminar bordas/fundo indesejado."""
    h, w, _ = img_np.shape
    margem_h = int(h * 0.2)
    margem_w = int(w * 0.2)
    recorte = img_np[margem_h:h-margem_h, margem_w:w-margem_w]
    return recorte

def processar_validacao(img_orig_path, img_teste_np):
    # Carrega a imagem original de referência
    img1 = cv2.imread(img_orig_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        return 0, 0, 0.0, "Erro: Não foi possível carregar a imagem original de referência."
    
    # Converte a imagem de teste do celular para escala de cinza
    img2 = cv2.cvtColor(img_teste_np, cv2.COLOR_RGB2GRAY)
    
    # Processo de extração de características ORB
    akaze = cv2.ORB_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0, 0, 0.0, "ALERTA! Poucos pontos encontrados ou imagem muito fora de foco."

    # Matcher com distância de Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Filtro Lowe's ratio
    bons = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            bons.append(m)
            
    total_possivel = max(len(kp1), len(kp2), 1)
    inliers_qtde = len(bons)
    correlacao = (inliers_qtde / total_possivel) * 100
    
    if correlacao > 15:
        status = "SELO ORIGINAL (Autêntico)"
    else:
        status = "ALERTA! SELO ADULTERADO OU FALSIFICADO"
        
    return inliers_qtde, total_possivel, correlacao, status

# 3. Interface Visual e Sidebar
st.title("🛡️ Validador de Relevo UV (PUF)")
st.write("Sistema de autenticação de selos físicos exclusivos baseados em micro-relevos de impressão UV.")

PASTA_ORIGINAIS = "selos_originais"
if not os.path.exists(PASTA_ORIGINAIS):
    os.makedirs(PASTA_ORIGINAIS)

arquivos_originais = [f for f in os.listdir(PASTA_ORIGINAIS) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

st.sidebar.header("Configuração do Servidor")
if arquivos_originais:
    selo_escolhido = st.sidebar.selectbox("Selecione o Selo de Referência (Original):", arquivos_originais)
    caminho_original = os.path.join(PASTA_ORIGINAIS, selo_escolhido)
else:
    st.sidebar.warning(f"A pasta '{PASTA_ORIGINAIS}' está vazia. Adicione a foto do selo original nela.")
    caminho_original = None

# 4. Envio de Imagem e Execução
st.write("### Envie a foto tirada pelo celular para validação")
arquivo_upload = st.file_uploader("Escolha a foto do selo de teste (JPG ou PNG)", type=["jpg", "jpeg", "png"])

if arquivo_upload is not None and caminho_original:
    image = Image.open(arquivo_upload)
    st.image(image, caption="Foto enviada do celular para auditoria", use_container_width=True)
    
    if st.button("Executar Auditoria de Relevo"):
        with st.spinner("Analisando micro-relevos tridimensionais..."):
            # Converte a imagem do Streamlit para formato OpenCV
            img_np = np.array(image)
            
            # ✂️ APLICA O RECORTE AUTOMÁTICO DO CENTRO ANTES DE VALIDAR
            img_np_processada = recortar_centro(img_np)
            
            # Roda a função de auditoria com a imagem já limpa de bordas
            inliers, total, correlacao, status = processar_validacao(caminho_original, img_np_processada)
            
            st.divider()
            st.subheader("Resultado da Auditoria:")
            
            if "ORIGINAL" in status:
                st.success(f"**STATUS:** {status}")
            else:
                st.error(f"**STATUS:** {status}")
                
            col1, col2 = st.columns(2)
            col1.metric("Pontos Validados (Inliers)", f"{inliers} de {total}")
            col2.metric("Índice de Correlação", f"{correlacao:.2f}%")
elif not arquivos_originais:
    st.info("Por favor, crie uma pasta chamada **selos_originais** no mesmo diretório do script e coloque ao menos uma foto padrão do selo lá.")
