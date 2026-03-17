<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&pause=1000&color=7C3AED&center=true&vCenter=true&width=600&lines=🤟+Tradutor+de+Libras;Reconhecimento+em+Tempo+Real;MediaPipe+%2B+Machine+Learning" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-0097A7?style=for-the-badge&logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

<br/>

> **Reconhecimento do alfabeto manual de Libras em tempo real**  
> direto no navegador, usando sua webcam — sem instalação, sem plugins.

<br/>

![demo](https://img.shields.io/badge/🚀_Demo_ao_vivo-Acessar-7C3AED?style=for-the-badge)

</div>

---

## ✨ Funcionalidades

- 🎥 **Detecção em tempo real** via webcam no navegador
- 🤖 **Machine Learning** treinado com seus próprios gestos
- 🔤 **Forma frases** letra por letra com estabilização inteligente
- 📊 **Top 5 probabilidades** exibidas ao vivo
- ⚡ **WebSocket** para comunicação de baixa latência
- 🐳 **Docker** pronto para produção
- 🔄 **CI/CD** automático com GitHub Actions + Render

---

## 🧠 Como Funciona

```
┌─────────────┐     frames JPEG      ┌──────────────────┐
│   Navegador  │ ──────────────────▶ │   FastAPI Backend │
│  (Webcam)   │ ◀────────────────── │   WebSocket /ws   │
└─────────────┘    letra + confiança └────────┬─────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │     MediaPipe      │
                                    │  21 landmarks/mão  │
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │  Random Forest /   │
                                    │  SVM / KNN /       │
                                    │  Ensemble          │
                                    │  (99.94% acurácia) │
                                    └───────────────────┘
```

---

## 🗂 Estrutura do Projeto

```
📦 Tradutor LIBRAS
├── 📁 .github/workflows/
│   └── deploy.yml          # CI/CD automático
├── 📁 backend/
│   └── main.py             # FastAPI + WebSocket
├── 📁 frontend/
│   └── index.html          # Interface web
├── 📁 data/
│   └── meus_dados.csv      # Dataset coletado
├── 📁 model/
│   ├── asl_classifier.pkl      # Modelo treinado
│   └── asl_label_encoder.pkl
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── ☁️  render.yaml
├── 📋 requirements.txt
├── 🎥 collect_data.py      # Coleta gestos pela webcam
└── 🏋️  train_asl.py         # Treina o modelo
```

---

## 🚀 Rodando Localmente

### Pré-requisitos
- Python 3.10
- Webcam

### 1. Clone o repositório
```bash
git clone https://github.com/SEU_USUARIO/tradutor-libras.git
cd tradutor-libras
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Colete seus gestos
```bash
python collect_data.py
```
> Uma janela abre com a câmera. Para cada letra: faça o sinal → pressione **ESPAÇO** → segure por 3s.
> **S** pula a letra, **Q** salva e sai.

### 4. Treine o modelo
```bash
python train_asl.py
```
> Testa automaticamente Random Forest, KNN, SVM e Ensemble — salva o melhor.

### 5. Inicie o app
```bash
python backend/main.py
```

Acesse **http://localhost:8000** 🎉

---

## 🐳 Rodando com Docker

```bash
# Sobe tudo com um comando
docker-compose up --build

# Acesse: http://localhost:8000
```

> ⚠️ Treine o modelo localmente antes — os arquivos `model/*.pkl` são montados como volume.

---

## ☁️ Deploy no Render

O deploy é **automático** a cada `git push` na branch `main`.

### Configuração inicial

1. Crie uma conta em [render.com](https://render.com)
2. Conecte seu repositório GitHub
3. Adicione os secrets no GitHub (**Settings → Secrets → Actions**):

| Secret | Onde encontrar |
|--------|---------------|
| `RENDER_API_KEY` | Render → Account Settings → API Keys |
| `RENDER_SERVICE_ID` | Render → Serviço → Settings → Service ID |

4. Faça um push — o CI/CD cuida do resto! ✅

### Fluxo CI/CD

```
git push origin main
       │
       ▼
  GitHub Actions
       │
       ├── ✅ Testes (imports, estrutura)
       │
       └── 🚀 Deploy automático no Render
```

---

## 📊 Performance do Modelo

| Modelo | Acurácia |
|--------|----------|
| Random Forest | ~99% |
| KNN (k=3) | ~98% |
| SVM (RBF) | ~99% |
| **Ensemble** | **99.94%** |

> Treinado com **24.000 amostras** — 1.000 por letra, 24 letras estáticas do alfabeto.

---

## 🔤 Letras Suportadas

```
A B C D E F G H I K L M N O P Q R S T U V W X Y
```
> J e Z são movimentos dinâmicos — não suportados nesta versão.

---

## 🛠 Tecnologias

| Camada | Tecnologia |
|--------|-----------|
| Detecção de mãos | MediaPipe Hands |
| Machine Learning | scikit-learn |
| Backend | FastAPI + WebSocket |
| Frontend | HTML + CSS + JS puro |
| Deploy | Render |
| CI/CD | GitHub Actions |
| Container | Docker |

---

## 📝 Próximos Passos

- [ ] Adicionar letras exclusivas do Libras (Ã, Õ, Ç)
- [ ] Reconhecimento de palavras completas
- [ ] Suporte a J e Z (movimentos dinâmicos)
- [ ] App mobile

---

<div align="center">

Feito com 🤟 por **Fernando**

</div>