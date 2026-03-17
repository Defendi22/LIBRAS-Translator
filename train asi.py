"""
=============================================================
  Treino OTIMIZADO — Classificador de Letras Libras
  Usa ensemble de modelos para máxima precisão
=============================================================
  Instale extras:
    pip install scikit-learn pandas numpy matplotlib seaborn joblib
  Rode:
    python train_asl.py
=============================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("=" * 55)
print("  Treino OTIMIZADO — Classificador Libras")
print("=" * 55)

CSV = "data/meus_dados.csv"
if not os.path.exists(CSV):
    print(f"❌ Arquivo não encontrado: {CSV}")
    exit(1)

df = pd.read_csv(CSV)
print(f"\n✅ {len(df)} amostras | {df['label'].nunique()} letras")

X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values

le    = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── Treinar múltiplos modelos e escolher o melhor ──
print("\n⏳ Testando modelos...")

modelos = {
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        ))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42,
        ))
    ]),
}

resultados = {}
for nome, modelo in modelos.items():
    print(f"  Treinando {nome}...")
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    resultados[nome] = (modelo, acc)
    print(f"  ✅ {nome}: {acc*100:.2f}%")

# ── Ensemble dos 3 melhores ──
print("\n⏳ Treinando Ensemble (VotingClassifier)...")
ensemble = VotingClassifier(
    estimators=[
        ("rf",  modelos["RandomForest"]),
        ("knn", modelos["KNN"]),
        ("svm", modelos["SVM"]),
    ],
    voting="soft",
    n_jobs=-1,
)
ensemble.fit(X_train, y_train)
acc_ensemble = accuracy_score(y_test, ensemble.predict(X_test))
resultados["Ensemble"] = (ensemble, acc_ensemble)
print(f"  ✅ Ensemble: {acc_ensemble*100:.2f}%")

# ── Escolher o melhor ──
melhor_nome = max(resultados, key=lambda k: resultados[k][1])
melhor_modelo, melhor_acc = resultados[melhor_nome]
print(f"\n🏆 Melhor modelo: {melhor_nome} ({melhor_acc*100:.2f}%)")

# ── Relatório completo ──
y_pred = melhor_modelo.predict(X_test)
print("\n📋 Relatório por letra:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Salvar ──
os.makedirs("model", exist_ok=True)
joblib.dump(melhor_modelo, "model/asl_classifier.pkl", compress=3)
joblib.dump(le,            "model/asl_label_encoder.pkl")
print(f"✅ Modelo salvo: model/asl_classifier.pkl")

# ── Comparativo de modelos ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Treino Otimizado — Libras", fontsize=14, fontweight="bold")

nomes = list(resultados.keys())
accs  = [resultados[n][1]*100 for n in nomes]
cores = ["#22c55e" if n == melhor_nome else "#64748b" for n in nomes]
axes[0].bar(nomes, accs, color=cores, edgecolor="white")
axes[0].set_ylim([min(accs)-2, 101])
axes[0].set_title("Comparativo de Modelos")
axes[0].set_ylabel("Acurácia (%)")
for i, (n, a) in enumerate(zip(nomes, accs)):
    axes[0].text(i, a+0.2, f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[1], cbar=False)
axes[1].set_title(f"Matriz de Confusão — {melhor_nome}")
axes[1].set_ylabel("Real"); axes[1].set_xlabel("Previsto")

plt.tight_layout()
plt.savefig("model/resultados.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Gráfico salvo em model/resultados.png")

print(f"""
{'='*55}
  ✅ TREINO CONCLUÍDO!
  Melhor modelo : {melhor_nome}
  Acurácia      : {melhor_acc*100:.2f}%
{'='*55}
🚀 Próximo passo: python backend/main.py
""")