import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 1. Configuração da página (DEVE SER O PRIMEIRO COMANDO DO STREAMLIT)
st.set_page_config(
    page_title="Validador de Selo UV - PUF", page_icon="🔍", layout="centered"
)


# 2. Funções auxiliares de processamento de imagem e vídeo dinâmico
def recortar_centro(img_np):
  """Recorta os 60% centrais da imagem para focar no selo e eliminar bordas/fundo indesejado."""
  h, w, _ = img_np.shape
  margem_h = int(h * 0.2)
  margem_w = int(w * 0.2)
  recorte = img_np[margem_h : h - margem_h, margem_w : w - margem_w]
  return recorte


def processar_validacao_frame(img_orig_path, img_teste_np):
  img1 = cv2.imread(img_orig_path, cv2.IMREAD_GRAYSCALE)
  if img1 is None or img_teste_np is None:
    return 0, 0, 0.0, "Erro de processamento"

  img2 = cv2.cvtColor(img_teste_np, cv2.COLOR_RGB2GRAY)

  orb = cv2.ORB_create()
  kp1, des1 = orb.detectAndCompute(img1, None)
  kp2, des2 = orb.detectAndCompute(img2, None)

  if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
    return 0, 0, 0.0, "Poucos pontos"

  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(des1, des2, k=2)

  bons = [m for m, n in matches if m.distance < 0.75 * n.distance]
  total_possivel = max(len(kp1), len(kp2), 1)
  correlacao = (len(bons) / total_possivel) * 100

  status = (
      "SELO ORIGINAL (Autêntico)"
      if correlacao > 15
      else "ALERTA! SELO ADULTERADO OU FALSIFICADO"
  )
  return len(bons), total_possivel, correlacao, status


def processar_validacao_video(img_orig_path, arquivo_video):
  tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
  tfile.write(arquivo_video.read())
  video_path = tfile.name

  cap = cv2.VideoCapture(video_path)
  melhor_correlacao = 0.0
  melhor_inliers = 0
  melhor_total = 0
  status_final = "ALERTA! SELO ADULTERADO OU FALSIFICADO"

  frame_count = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Analisa a cada 3 quadros para manter o processamento fluido e rápido
    frame_count += 1
    if frame_count % 3 != 0:
      continue

    # Converte BGR do OpenCV para RGB e aplica o recorte central automático
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_processado = recortar_centro(frame_rgb)

    inliers, total, correlacao, status = processar_validacao_frame(
        img_orig_path, frame_processado
    )

    if correlacao > melhor_correlacao:
      melhor_correlacao = correlacao
      melhor_inliers = inliers
      melhor_total = total
      if "ORIGINAL" in status:
        status_final = status

  cap.release()
  try:
    os.unlink(video_path)
  except:
    pass

  return melhor_inliers, melhor_total, melhor_correlacao, status_final


# 3. Interface Visual e Configurações de Barra Lateral
st.title("🛡️ Validador Dinâmico de Relevo UV (PUF)")
st.write(
    "Sistema avançado de autenticação de selos baseados em micro-relevos de"
    " impressão UV com compensação dinâmica de luz."
)

PASTA_ORIGINAIS = "selos_originais"
if not os.path.exists(PASTA_ORIGINAIS):
  os.makedirs(PASTA_ORIGINAIS)

arquivos_originais = [
    f
    for f in os.listdir(PASTA_ORIGINAIS)
    if f.lower().endswith(("png", "jpg", "jpeg"))
]

st.sidebar.header("Configuração do Servidor")
if arquivos_originais:
  selo_escolhido = st.sidebar.selectbox(
      "Selecione o Selo de Referência (Original):", arquivos_originais
  )
  caminho_original = os.path.join(PASTA_ORIGINAIS, selo_escolhido)
else:
  st.sidebar.warning(
      f"A pasta '{PASTA_ORIGINAIS}' está vazia. Adicione a foto do selo"
      " original nela."
  )
  caminho_original = None

# 4. Seção de Envio de Vídeo e Execução da Auditoria
st.write("### Envie um mini-vídeo do selo em movimento")
st.info(
    "💡 **Dica de uso:** Grave um vídeo rápido (2 a 3 segundos) passando o"
    " celular suavemente sobre o selo. O algoritmo vai varrer os quadros e"
    " encontrar o momento ideal onde o reflexo cruza perfeitamente o relevo."
)

arquivo_upload = st.file_uploader(
    "Escolha o arquivo de vídeo (MP4, MOV ou AVI)", type=["mp4", "mov", "avi"]
)

if arquivo_upload is not None and caminho_original:
  st.video(arquivo_upload)

  if st.button("Executar Auditoria Dinâmica por Vídeo"):
    with st.spinner("Analisando quadros e varrendo os micro-relevos PUF..."):
      inliers, total, correlacao, status = processar_validacao_video(
          caminho_original, arquivo_upload
      )

      st.divider()
      st.subheader("Resultado da Auditoria:")

      if "ORIGINAL" in status:
        st.success(f"**STATUS:** {status}")
      else:
        st.error(f"**STATUS:** {status}")

      col1, col2 = st.columns(2)
      col1.metric("Melhor Quadro - Pontos Validados", f"{inliers} de {total}")
      col2.metric("Índice de Correlação Máxima", f"{correlacao:.2f}%")

elif not arquivos_originais:
  st.info(
      "Por favor, certifique-se de que a pasta **selos_originais** possui ao"
      " menos uma imagem de referência cadastrada no repositório."
  )
