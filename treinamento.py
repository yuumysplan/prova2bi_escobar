from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_DADOS = BASE_DIR / "HousingData.csv"
ARQUIVO_MODELO = BASE_DIR / "modelo_kmeans.joblib"
ARQUIVO_COTOVELO = BASE_DIR / "cotovelo.png"

COLS_BINARIAS = ["CHAS"]
COLS_CATEGORICAS = ["RAD"]
COLS_QUANTITATIVAS = [
    "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS",
    "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
]
ORDEM_ORIGINAL = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
]

K_MAX = 10
SEMENTE = 42


def carregar_dados(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho)


def imputar(df: pd.DataFrame):
    valores = {}
    for c in COLS_QUANTITATIVAS:
        valores[c] = float(df[c].median())
    for c in COLS_BINARIAS + COLS_CATEGORICAS:
        valores[c] = df[c].mode().iloc[0]
    return df.fillna(value=valores), valores


def ajustar_normalizadores(df: pd.DataFrame):
    scaler = MinMaxScaler().fit(df[COLS_QUANTITATIVAS])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        df[COLS_CATEGORICAS].astype(int)
    )
    return scaler, encoder


def normalizar(df: pd.DataFrame, scaler, encoder) -> np.ndarray:
    quant = scaler.transform(df[COLS_QUANTITATIVAS])
    binar = df[COLS_BINARIAS].astype(float).to_numpy()
    categ = encoder.transform(df[COLS_CATEGORICAS].astype(int))
    return np.hstack([quant, binar, categ])


def metodo_cotovelo(X: np.ndarray, k_max: int = K_MAX):
    ks = np.arange(1, k_max + 1)
    inercias = np.array(
        [KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit(X).inertia_ for k in ks]
    )

    p1 = np.array([ks[0], inercias[0]])
    p2 = np.array([ks[-1], inercias[-1]])
    direcao = (p2 - p1) / np.linalg.norm(p2 - p1)
    distancias = []
    for k, ine in zip(ks, inercias):
        v = np.array([k, ine]) - p1
        proj = np.dot(v, direcao) * direcao
        distancias.append(np.linalg.norm(v - proj))
    distancias = np.array(distancias)
    k_otimo = int(ks[np.argmax(distancias)])
    return ks, inercias, distancias, k_otimo


def plotar_cotovelo(ks, inercias, k_otimo, caminho: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inercias, "o-", color="#2a6fdb")
    plt.plot([ks[0], ks[-1]], [inercias[0], inercias[-1]], "--", color="gray",
             label="reta entre extremos")
    plt.axvline(k_otimo, color="#d9534f", linestyle=":", label=f"k ótimo = {k_otimo}")
    plt.title("Método do cotovelo — K-Means (Boston Housing)")
    plt.xlabel("Número de grupos (k)")
    plt.ylabel("Inércia (soma das distâncias² aos centroides)")
    plt.xticks(ks)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()


def main():
    df = carregar_dados(ARQUIVO_DADOS)
    print(f"Base carregada: {df.shape[0]} registros, {df.shape[1]} atributos")
    print(f"Valores ausentes: {int(df.isna().sum().sum())}")

    df, valores_imputacao = imputar(df)
    scaler, encoder = ajustar_normalizadores(df)
    X = normalizar(df, scaler, encoder)
    print(f"Matriz normalizada: {X.shape} "
          f"({len(COLS_QUANTITATIVAS)} quantitativas + {len(COLS_BINARIAS)} binária + "
          f"{X.shape[1] - len(COLS_QUANTITATIVAS) - len(COLS_BINARIAS)} colunas one-hot de RAD)")

    ks, inercias, distancias, k_otimo = metodo_cotovelo(X)
    print("\nMétodo do cotovelo:")
    print(f"{'k':>3} | {'inércia':>10} | {'dist. à reta':>12}")
    for k, i, d in zip(ks, inercias, distancias):
        marca = "  <- cotovelo" if k == k_otimo else ""
        print(f"{k:>3} | {i:>10.3f} | {d:>12.3f}{marca}")
    plotar_cotovelo(ks, inercias, k_otimo, ARQUIVO_COTOVELO)
    print(f"Gráfico salvo em: {ARQUIVO_COTOVELO.name}")

    modelo = KMeans(n_clusters=k_otimo, n_init=10, random_state=SEMENTE).fit(X)
    tamanhos = np.bincount(modelo.labels_)
    print(f"\nK-Means treinado com k = {k_otimo}. Tamanho dos grupos: {tamanhos.tolist()}")

    joblib.dump(
        {
            "modelo": modelo,
            "scaler": scaler,
            "encoder": encoder,
            "valores_imputacao": valores_imputacao,
            "cols_quantitativas": COLS_QUANTITATIVAS,
            "cols_binarias": COLS_BINARIAS,
            "cols_categoricas": COLS_CATEGORICAS,
            "ordem_original": ORDEM_ORIGINAL,
            "k": k_otimo,
        },
        ARQUIVO_MODELO,
    )
    print(f"Modelo salvo em: {ARQUIVO_MODELO.name}")


if __name__ == "__main__":
    main()
