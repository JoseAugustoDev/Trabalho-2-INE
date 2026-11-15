from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALVO = "Revenue"
SEMENTE = 42
NUMERICAS = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAS = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]


def carregar_dados(caminho):
    dados = pd.read_csv(Path(caminho))
    esperadas = set(NUMERICAS + CATEGORICAS + [ALVO])
    if set(dados.columns) != esperadas or dados.columns.duplicated().any():
        raise ValueError("O CSV deve conter exatamente as colunas esperadas, sem nomes repetidos.")
    for coluna in NUMERICAS + CATEGORICAS + [ALVO]:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="raise")
    if dados[ALVO].isna().any() or not dados[ALVO].isin([0, 1]).all():
        raise ValueError("Revenue deve conter apenas 0 ou 1, sem valores ausentes.")
    for coluna, valores in {"Month": range(1, 13), "VisitorType": [0, 1, 2], "Weekend": [0, 1]}.items():
        if not dados[coluna].dropna().isin(valores).all():
            raise ValueError(f"A coluna {coluna} contém códigos inesperados.")
    if (dados[NUMERICAS] < 0).any().any():
        raise ValueError("Há valores negativos nos atributos numéricos; investigue antes de prosseguir.")
    for coluna in ["BounceRates", "ExitRates", "SpecialDay"]:
        if (dados[coluna] > 1).any():
            raise ValueError(f"{coluna} contém valores acima de 1; investigue antes de prosseguir.")
    return dados


def preparar_dados(dados):
    """Remove duplicatas exatas antes dos folds e separa preditores e alvo."""
    dados = dados.drop_duplicates().reset_index(drop=True)
    return dados[NUMERICAS + CATEGORICAS].copy(), dados[ALVO].astype(int).copy()


def criar_preprocessador():
    """Transformações ajustadas exclusivamente no treinamento de cada fold."""
    numerico = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("padronizacao", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("codificacao", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numericas", numerico, NUMERICAS),
        ("categoricas", categorico, CATEGORICAS),
    ])


def criar_particoes(X, y):
    """A mesma lista deve ser utilizada para todos os modelos, com X e y na mesma ordem."""
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE).split(X, y))
