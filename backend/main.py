"""
=============================================================
  Libras Translator — Backend OTIMIZADO
  - Inferência direta sem sleep
  - Buffer de frames para estabilização rápida
  - Landmarks normalizados igual ao collect_data.py
=============================================================
  Para rodar:
    python backend/main.py
=============================================================
"""

import os, sys, json, base64, asyncio
import numpy as np
import cv2
import joblib
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Carregar modelo ───────────────────────
print("Carregando modelo...")
clf = joblib.load(os.path.join(ROOT, "model", "asl_classifier.pkl"))
le  = joblib.load(os.path.join(ROOT, "model", "asl_label_encoder.pkl"))
print(f"✅ {len(le.classes_)} letras prontas: {list(le.classes_)}")

# ── MediaPipe com configurações otimizadas ─
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,           # 0=rápido, 1=preciso
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ── App ───────────────────────────────────
app = FastAPI(title="Libras Translator")
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

# ── Extração IDÊNTICA ao collect_data.py ──
def extrair_landmarks(hand_landmarks):
    lms = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wx, wy, wz = lms[0]
    features = []
    for (x, y, z) in lms:
        features += [round(x-wx, 6), round(y-wy, 6), round(z-wz, 6)]
    return features

# ── Buffer de estabilização ───────────────
class Buffer:
    """
    Mantém as últimas N predições.
    Retorna a letra mais votada se tiver maioria clara.
    """
    def __init__(self, n=5):
        self.n       = n
        self.letras  = []
        self.confias = []

    def add(self, letra, conf):
        self.letras.append(letra)
        self.confias.append(conf)
        if len(self.letras) > self.n:
            self.letras.pop(0)
            self.confias.pop(0)

    def get(self):
        if len(self.letras) < 2:
            return None, 0.0
        votos    = Counter(self.letras)
        vencedor, count = votos.most_common(1)[0]
        # Exige maioria simples (>50%)
        if count / len(self.letras) < 0.5:
            return None, 0.0
        conf_media = np.mean([c for l, c in zip(self.letras, self.confias) if l == vencedor])
        return vencedor, round(float(conf_media) * 100, 1)

    def clear(self):
        self.letras.clear()
        self.confias.clear()

# ── WebSocket ─────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    buf = Buffer(n=5)
    print("🔌 Cliente conectado")

    try:
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)

            if msg.get("type") != "frame":
                continue

            # Decodifica frame
            raw = base64.b64decode(msg["image"].split(",")[-1])
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Reduz resolução para acelerar MediaPipe
            h, w = img.shape[:2]
            if w > 480:
                scale = 480 / w
                img   = cv2.resize(img, (480, int(h * scale)))

            # Detecta mão
            rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb)

            if not results.multi_hand_landmarks:
                buf.clear()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "hand_detected": False,
                }))
                continue

            # Extrai features — idêntico ao collect_data.py
            features = extrair_landmarks(results.multi_hand_landmarks[0])
            X        = np.array(features, dtype=np.float32).reshape(1, -1)

            # Classifica
            proba  = clf.predict_proba(X)[0]
            idx    = int(np.argmax(proba))
            letra  = le.inverse_transform([idx])[0]
            conf   = float(proba[idx])

            # Adiciona ao buffer
            buf.add(letra, conf)
            letra_stable, conf_stable = buf.get()

            # Top 5
            top5_idx = np.argsort(proba)[::-1][:5]
            top5 = [
                {"letter": le.inverse_transform([i])[0],
                 "confidence": round(float(proba[i]) * 100, 1)}
                for i in top5_idx
            ]

            if letra_stable and conf_stable >= 65:
                await websocket.send_text(json.dumps({
                    "type":          "prediction",
                    "hand_detected":  True,
                    "letter":         letra_stable,
                    "confidence":     conf_stable,
                    "raw_letter":     letra,
                    "top5":           top5,
                }))
            else:
                # Envia mesmo assim para mostrar o top5 em tempo real
                await websocket.send_text(json.dumps({
                    "type":          "detecting",
                    "hand_detected":  True,
                    "letter":         letra,
                    "confidence":     round(conf * 100, 1),
                    "top5":           top5,
                }))

    except WebSocketDisconnect:
        print("🔌 Desconectado")
    except Exception as e:
        print(f"❌ {e}")
        try: await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except: pass

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 http://localhost:8000\n")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        loop="asyncio",
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )