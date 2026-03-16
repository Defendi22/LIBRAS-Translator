"""
=============================================================
  Treino — Classificador de Letras Libras
  Entrada : data/meus_dados.csv
  Saída   : model/asl_classifier.pkl
=============================================================
  Rode após collect_data.py:
    python train_asl.py
=============================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("=" * 50)
print("  Treino — Classificador de Letras Libras")
print("=" * 50)

# ── Carregar dados ────────────────────────
CSV = "data/meus_dados.csv"
if not os.path.exists(CSV):
    print(f"❌ Arquivo não encontrado: {CSV}")
    print("   Rode primeiro: python collect_data.py")
    exit(1)

df = pd.read_csv(CSV)
print(f"\n✅ {len(df)} amostras | {df['label'].nunique()} letras")
print(f"   Letras: {sorted(df['label'].unique())}")
print(f"   Amostras por letra:\n{df['label'].value_counts().sort_index().to_string()}")

X = df.drop("label", axis=1).values
y = df["label"].values

# ── Codificar labels ──────────────────────
le    = LabelEncoder()
y_enc = le.fit_transform(y)

# ── Split ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\n✅ Treino: {len(X_train)} | Teste: {len(X_test)}")

# ── Treinar ───────────────────────────────
print("\n⏳ Treinando Random Forest...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# ── Avaliar ───────────────────────────────
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n✅ Acurácia: {acc*100:.2f}%")
print("\n📋 Por letra:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Salvar ────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/asl_classifier.pkl")
joblib.dump(le,  "model/asl_label_encoder.pkl")
print("✅ Modelo salvo em model/asl_classifier.pkl")

# ── Matriz de confusão ────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Matriz de Confusão (acurácia: {acc*100:.1f}%)")
plt.ylabel("Real"); plt.xlabel("Previsto")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Gráfico salvo em model/confusion_matrix.png")

print(f"""
{'='*50}
  ✅ TREINO CONCLUÍDO!
  Acurácia : {acc*100:.2f}%
  Modelo   : model/asl_classifier.pkl

🚀 Próximo passo: python backend/main.py
{'='*50}
""")