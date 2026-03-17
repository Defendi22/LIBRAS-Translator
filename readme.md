# 🤟 Tradutor de Libras em Tempo Real

Reconhecimento do alfabeto manual de Libras via webcam usando MediaPipe + Machine Learning.

---

## 🗂 Estrutura do Projeto

```
Tradutor LIBRAS/
├── .github/
│   └── workflows/
│       └── deploy.yml      ← CI/CD automático
├── backend/
│   └── main.py             ← FastAPI + WebSocket
├── frontend/
│   └── index.html          ← Interface web
├── data/
│   └── meus_dados.csv      ← Gerado pelo collect_data.py
├── model/
│   ├── asl_classifier.pkl      ← Gerado pelo train_asl.py
│   └── asl_label_encoder.pkl
├── collect_data.py         ← Coleta gestos pela webcam
├── train_asl.py            ← Treina o modelo
├── Dockerfile
├── docker-compose.yml
├── render.yaml
└── requirements.txt
```

---

## 🚀 Rodando Localmente

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Coletar dados (primeira vez)
```bash
python collect_data.py
# Siga as instruções na tela para cada letra
```

### 3. Treinar o modelo
```bash
python train_asl.py
```

### 4. Iniciar o app
```bash
python backend/main.py
# Acesse: http://localhost:8000
```

---

## 🐳 Rodando com Docker

```bash
# Build e start
docker-compose up --build

# Acesse: http://localhost:8000
```

> ⚠️ O modelo precisa estar treinado antes do Docker.
> Os arquivos em `model/` são montados como volume.

---

## ☁️ Deploy no Render

### Pré-requisitos
1. Conta no [Render](https://render.com)
2. Conta no [Docker Hub](https://hub.docker.com)
3. Repositório no GitHub com o projeto

### Configurar secrets no GitHub
Vá em **Settings → Secrets → Actions** e adicione:

| Secret | Valor |
|--------|-------|
| `DOCKER_USERNAME` | Seu usuário do Docker Hub |
| `DOCKER_PASSWORD` | Sua senha do Docker Hub |
| `RENDER_API_KEY` | API Key do Render (dashboard → Account Settings) |
| `RENDER_SERVICE_ID` | ID do serviço no Render (dashboard → serviço → Settings) |

### Deploy automático
Após configurar os secrets, qualquer push na branch `main` irá:
1. ✅ Rodar os testes
2. 🐳 Fazer build e push da imagem Docker
3. 🚀 Fazer deploy automático no Render

### Primeiro deploy manual
1. Acesse [render.com](https://render.com)
2. Clique em **New → Web Service**
3. Conecte seu repositório GitHub
4. O Render detecta o `render.yaml` automaticamente
5. Clique em **Deploy**

---

