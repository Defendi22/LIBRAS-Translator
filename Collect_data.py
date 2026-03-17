"""
=============================================================
  Coletor de Dados — Alfabeto Libras
  Usa a webcam para capturar landmarks de cada letra
=============================================================
  Como usar:
    python collect_data.py

  Para cada letra:
    - O script mostra a letra na tela
    - Pressione ESPAÇO para começar a gravar
    - Faça o sinal e mantenha por ~3 segundos
    - O script captura 50 amostras automaticamente
    - Repita para a próxima letra

  Teclas:
    ESPAÇO → iniciar captura da letra atual
    S      → pular letra atual
    Q      → sair e salvar
=============================================================
"""

import cv2
import csv
import os
import mediapipe as mp
import numpy as np

# ── Config ───────────────────────────────
LETRAS        = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 letras estáticas
AMOSTRAS_POR_LETRA = 1000
OUTPUT_CSV    = "data/meus_dados.csv"

os.makedirs("data", exist_ok=True)

# ── MediaPipe ────────────────────────────
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
hands     = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6
)

# ── CSV ──────────────────────────────────
# Verifica se já existe arquivo (para continuar de onde parou)
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
header.append("label")

letras_ja_coletadas = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            letras_ja_coletadas.add(row["label"])
    print(f"✅ Arquivo existente encontrado. Letras já coletadas: {sorted(letras_ja_coletadas)}")
    file_mode = "a"  # append
else:
    file_mode = "w"  # novo arquivo

# ── Extração de landmarks ─────────────────
def extrair_landmarks(hand_landmarks):
    lms = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wx, wy, wz = lms[0]
    features = []
    for (x, y, z) in lms:
        features += [round(x-wx,6), round(y-wy,6), round(z-wz,6)]
    return features

# ── Interface visual ─────────────────────
def desenhar_ui(frame, letra, status, coletadas, total, mao_detectada):
    h, w = frame.shape[:2]

    # Fundo escuro no topo
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 100), (15,17,23), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Letra grande
    cv2.putText(frame, letra, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, (139, 92, 246), 4)

    # Status
    cor_status = (34,197,94) if status == "GRAVANDO" else (251,191,36) if status == "PRONTO" else (148,163,184)
    cv2.putText(frame, status, (160, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor_status, 2)

    # Progresso
    barra_w = int((coletadas / total) * (w - 40))
    cv2.rectangle(frame, (20, 110), (w-20, 126), (45,50,65), -1)
    if barra_w > 0:
        cv2.rectangle(frame, (20, 110), (20+barra_w, 126), (139,92,246), -1)
    cv2.putText(frame, f"{coletadas}/{total} amostras", (20, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148,163,184), 1)

    # Indicador de mão
    cor_mao = (34,197,94) if mao_detectada else (239,68,68)
    cv2.circle(frame, (w-30, 30), 12, cor_mao, -1)

    # Instruções
    instrucoes = "ESPACO=gravar  S=pular  Q=sair"
    cv2.putText(frame, instrucoes, (20, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (71,85,105), 1)

    return frame

# ── Loop principal ────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam não encontrada!")
    exit(1)

print("\n" + "="*50)
print("  🤟 Coletor de Dados — Libras")
print("="*50)
print(f"  Letras a coletar : {LETRAS}")
print(f"  Amostras/letra   : {AMOSTRAS_POR_LETRA}")
print(f"  Saída            : {OUTPUT_CSV}")
print("="*50)
print("\n  ESPAÇO = iniciar gravação")
print("  S      = pular letra")
print("  Q      = sair\n")

with open(OUTPUT_CSV, file_mode, newline="") as csvfile:
    writer = csv.writer(csvfile)
    if file_mode == "w":
        writer.writerow(header)

    for letra in LETRAS:
        if letra in letras_ja_coletadas:
            print(f"  ⏭  {letra} — já coletada, pulando...")
            continue

        print(f"\n  👉 Prepare o sinal para a letra: {letra}")
        print(f"     Pressione ESPAÇO para começar a gravar...")

        coletadas   = 0
        gravando    = False
        mao_ok      = False

        while coletadas < AMOSTRAS_POR_LETRA:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            mao_ok = res.multi_hand_landmarks is not None

            # Desenha landmarks se detectou mão
            if mao_ok:
                mp_draw.draw_landmarks(
                    frame,
                    res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(139,92,246), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(99,102,241), thickness=2),
                )

            # Grava se estiver no modo gravação e mão detectada
            if gravando and mao_ok:
                features = extrair_landmarks(res.multi_hand_landmarks[0])
                writer.writerow(features + [letra])
                csvfile.flush()
                coletadas += 1

            status = "GRAVANDO" if gravando else ("PRONTO — pressione ESPACO" if mao_ok else "Mostre sua mão...")
            frame  = desenhar_ui(frame, letra, status, coletadas, AMOSTRAS_POR_LETRA, mao_ok)
            cv2.imshow("Coletor de Dados — Libras", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n  ⏹  Saindo...")
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                print(f"\n✅ Dados salvos em {OUTPUT_CSV}")
                exit(0)
            elif key == ord(' '):
                if mao_ok:
                    gravando = True
                    print(f"  🔴 Gravando {letra}...")
                else:
                    print("  ⚠️  Nenhuma mão detectada! Posicione a mão primeiro.")
            elif key == ord('s'):
                print(f"  ⏭  Pulando letra {letra}...")
                break

        if coletadas >= AMOSTRAS_POR_LETRA:
            print(f"  ✅ {letra} — {coletadas} amostras coletadas!")
            gravando = False

cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"\n{'='*50}")
print(f"  ✅ COLETA CONCLUÍDA!")
print(f"  Arquivo: {OUTPUT_CSV}")
print(f"\n🚀 Próximo passo: python train_asl.py")
print(f"{'='*50}\n")