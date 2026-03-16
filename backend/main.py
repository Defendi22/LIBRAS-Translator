"""
=============================================================
  Libras Translator — Backend v2
  Modelo: Random Forest + MediaPipe Landmarks
=============================================================
  Para rodar:
    python backend/main.py
  Acesse: http://localhost:8000
=============================================================
"""

import os, sys, json, base64, asyncio
import numpy as np
import cv2
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Carregar modelo Random Forest ────────
print("Carregando modelo ASL...")
clf = joblib.load(os.path.join(ROOT, "model", "asl_classifier.pkl"))
le  = joblib.load(os.path.join(ROOT, "model", "asl_label_encoder.pkl"))
print(f"✅ Pronto — {len(le.classes_)} letras: {list(le.classes_)}")

# ── MediaPipe ────────────────────────────
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# ── App ──────────────────────────────────
app = FastAPI(title="Libras Translator v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND = os.path.join(ROOT, "frontend")
if os.path.exists(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/")
async def root():
    p = os.path.join(FRONTEND, "index.html")
    return HTMLResponse(open(p, encoding="utf-8").read()) if os.path.exists(p) else HTMLResponse("<h2>Frontend não encontrado.</h2>")

@app.get("/health")
async def health():
    return {"status": "ok", "letters": list(le.classes_)}

# ── Extração de landmarks ─────────────────
def extract_landmarks(hand_landmarks):
    """Extrai 63 features normalizadas pelo pulso."""
    lms = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist_x, wrist_y, wrist_z = lms[0]
    features = []
    for (x, y, z) in lms:
        features += [x - wrist_x, y - wrist_y, z - wrist_z]
    return np.array(features).reshape(1, -1)

# ── Suavizador de predições ───────────────
class PredictionSmoother:
    """
    Acumula N predições consecutivas e retorna
    a letra mais votada — evita flickering.
    """
    def __init__(self, window=7):
        self.window  = window
        self.history = []

    def add(self, letter, proba):
        self.history.append((letter, proba))
        if len(self.history) > self.window:
            self.history.pop(0)

    def get_stable(self):
        if not self.history:
            return None, 0.0
        # Vota na letra mais frequente
        from collections import Counter
        votes  = Counter(h[0] for h in self.history)
        winner = votes.most_common(1)[0][0]
        # Confiança média das ocorrências do vencedor
        conf   = np.mean([h[1] for h in self.history if h[0] == winner])
        return winner, round(float(conf) * 100, 1)

# ── WebSocket ────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    smoother = PredictionSmoother(window=7)
    print("🔌 Cliente conectado")

    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            if msg.get("type") != "frame":
                continue

            # Decodifica frame
            img = cv2.imdecode(
                np.frombuffer(base64.b64decode(msg["image"].split(",")[-1]), np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                continue

            # Detecta mão
            results = hands_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                smoother.history.clear()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "hand_detected": False,
                    "message": "Nenhuma mão detectada"
                }))
                continue

            # Extrai landmarks e classifica
            features = extract_landmarks(results.multi_hand_landmarks[0])
            proba    = clf.predict_proba(features)[0]
            idx      = int(np.argmax(proba))
            letter   = le.inverse_transform([idx])[0]
            conf     = float(proba[idx])

            # Suaviza predição
            smoother.add(letter, conf)
            stable_letter, stable_conf = smoother.get_stable()

            # Top 5 letras mais prováveis
            top5_idx = np.argsort(proba)[::-1][:5]
            top5 = [
                {"letter": le.inverse_transform([i])[0],
                 "confidence": round(float(proba[i]) * 100, 1)}
                for i in top5_idx
            ]

            await websocket.send_text(json.dumps({
                "type":         "prediction",
                "hand_detected": True,
                "letter":        stable_letter,
                "confidence":    stable_conf,
                "raw_letter":    letter,
                "raw_confidence": round(conf * 100, 1),
                "top5":          top5,
            }))

            await asyncio.sleep(0.03)  # ~30fps

    except WebSocketDisconnect:
        print("🔌 Desconectado")
    except Exception as e:
        print(f"❌ {e}")
        try: await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except: pass

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)