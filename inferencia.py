import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_MODELO = BASE_DIR / "modelo_kmeans.joblib"

DESCRICOES = {
    "CRIM": "taxa de criminalidade per capita",
    "ZN": "% de terrenos residenciais > 25.000 pés²",
    "INDUS": "% de área comercial não varejista",
    "CHAS": "margeia o rio Charles (1 = sim, 0 = não)",
    "NOX": "concentração de óxidos nítricos (ppm/10)",
    "RM": "nº médio de cômodos",
    "AGE": "% de imóveis construídos antes de 1940",
    "DIS": "distância ponderada a centros de emprego",
    "RAD": "índice de acesso a rodovias (1-8 ou 24)",
    "TAX": "imposto sobre propriedade (por US$ 10 mil)",
    "PTRATIO": "razão aluno/professor",
    "B": "1000(Bk - 0,63)²",
    "LSTAT": "% de população de baixa renda",
    "MEDV": "valor mediano do imóvel (US$ mil)",
}

EXEMPLO = {
    "CRIM": 0.05, "ZN": 20.0, "INDUS": 4.0, "CHAS": 0, "NOX": 0.45,
    "RM": 6.8, "AGE": 40.0, "DIS": 5.5, "RAD": 4, "TAX": 280.0,
    "PTRATIO": 17.0, "B": 395.0, "LSTAT": 6.0, "MEDV": 28.0,
}


def carregar_artefatos(caminho: Path = ARQUIVO_MODELO) -> dict:
    if not caminho.exists():
        raise SystemExit(f"Modelo não encontrado em {caminho}. Rode antes: python treinamento.py")
    return joblib.load(caminho)


def ler_interativo(art: dict) -> dict:
    print("Informe os dados da nova residência (Enter vazio = valor padrão da base):")
    dados = {}
    for col in art["ordem_original"]:
        padrao = art["valores_imputacao"][col]
        txt = input(f"  {col:<8} ({DESCRICOES[col]}) [{padrao}]: ").strip().replace(",", ".")
        dados[col] = float(txt) if txt else padrao
    return dados


def validar(dados: dict, art: dict) -> pd.DataFrame:
    faltantes = [c for c in art["ordem_original"] if c not in dados]
    if faltantes:
        print(f"Aviso: atributos ausentes preenchidos com o valor padrão do treino: {faltantes}")
    linha = {c: dados.get(c, art["valores_imputacao"][c]) for c in art["ordem_original"]}
    df = pd.DataFrame([linha])

    if int(df.at[0, "CHAS"]) not in (0, 1):
        raise SystemExit("CHAS deve ser 0 ou 1.")
    categorias_rad = art["encoder"].categories_[0].tolist()
    if int(df.at[0, "RAD"]) not in categorias_rad:
        print(f"Aviso: RAD={int(df.at[0, 'RAD'])} não visto no treino "
              f"(válidos: {categorias_rad}); será ignorado na distância.")
    return df


def normalizar(df: pd.DataFrame, art: dict) -> np.ndarray:
    quant = art["scaler"].transform(df[art["cols_quantitativas"]])
    binar = df[art["cols_binarias"]].astype(float).to_numpy()
    categ = art["encoder"].transform(df[art["cols_categoricas"]].astype(int))
    return np.hstack([quant, binar, categ])


def desnormalizar_centroide(centroide: np.ndarray, art: dict) -> dict:
    nq, nb = len(art["cols_quantitativas"]), len(art["cols_binarias"])
    quant = art["scaler"].inverse_transform(centroide[:nq].reshape(1, -1))[0]
    binar = centroide[nq:nq + nb]
    categ = centroide[nq + nb:]

    resultado = dict(zip(art["cols_quantitativas"], quant))
    for col, p in zip(art["cols_binarias"], binar):
        resultado[col] = f"{int(round(p))}  (proporção com 1: {p:.0%})"
    cats = art["encoder"].categories_[0]
    idx = int(np.argmax(categ))
    resultado["RAD"] = f"{int(cats[idx])}  (predominante: {categ[idx]:.0%} do grupo)"
    return {c: resultado[c] for c in art["ordem_original"]}


def inferir(dados: dict, art: dict | None = None):
    art = art or carregar_artefatos()
    df = validar(dados, art)
    X = normalizar(df, art)
    grupo = int(art["modelo"].predict(X)[0])
    distancia = float(np.linalg.norm(X[0] - art["modelo"].cluster_centers_[grupo]))
    centroide = desnormalizar_centroide(art["modelo"].cluster_centers_[grupo], art)
    return df.iloc[0].to_dict(), grupo, distancia, centroide


def formatar(v):
    if isinstance(v, (float, np.floating)):
        return f"{round(float(v), 4) + 0.0:.4f}"
    return str(v)


def main():
    p = argparse.ArgumentParser(description="Inferência de grupo (K-Means) para nova residência")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--json", help="dados da residência em JSON")
    g.add_argument("--arquivo", help="caminho de um arquivo JSON com os dados")
    g.add_argument("--exemplo", action="store_true", help="usa uma residência de exemplo")
    args = p.parse_args()

    art = carregar_artefatos()
    if args.json:
        dados = json.loads(args.json)
    elif args.arquivo:
        dados = json.loads(Path(args.arquivo).read_text(encoding="utf-8"))
    elif args.exemplo:
        dados = EXEMPLO
    else:
        dados = ler_interativo(art)

    entrada, grupo, distancia, centroide = inferir(dados, art)

    print(f"\nNova residência pertence ao GRUPO {grupo} (de {art['k']} grupos)")
    print(f"Distância ao centroide (espaço normalizado): {distancia:.4f}\n")
    print(f"{'Atributo':<9} | {'Nova residência':>16} | Centroide do grupo {grupo} (desnormalizado)")
    print("-" * 80)
    for col in art["ordem_original"]:
        print(f"{col:<9} | {formatar(entrada[col]):>16} | {formatar(centroide[col])}")


if __name__ == "__main__":
    main()
