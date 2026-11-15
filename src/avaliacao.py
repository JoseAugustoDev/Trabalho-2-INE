import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def avaliar_modelo(modelo, X, y, particoes):

    if len(X) != len(y) or len(y) == 0:
        raise ValueError("X e y devem ter o mesmo número não nulo de linhas.")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("O alvo deve conter apenas 0 e 1, sem valores ausentes.")

    particoes = list(particoes)
    contagem = np.zeros(len(y), dtype=int)
    for treino, teste in particoes:
        treino, teste = np.asarray(treino), np.asarray(teste)
        for indices in (treino, teste):
            if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or len(indices) == 0 or (indices < 0).any() or (indices >= len(y)).any()
                    or len(np.unique(indices)) != len(indices)):
                raise ValueError("Cada partição deve conter índices inteiros, válidos e sem repetições.")
        if np.intersect1d(treino, teste).size:
            raise ValueError("Treino e validação não podem compartilhar registros no mesmo fold.")
        contagem[teste] += 1
    if not np.all(contagem == 1):
        raise ValueError("Cada registro deve aparecer exatamente uma vez na validação.")

    resultados = []
    previsoes = np.empty(len(y), dtype=int)

    for fold, (treino, teste) in enumerate(particoes, 1):
        estimador = clone(modelo)

        X_treino = X.iloc[treino]
        X_teste = X.iloc[teste]

        y_treino = y.iloc[treino]
        y_teste = y.iloc[teste]

        estimador.fit(X_treino, y_treino)
        y_pred = estimador.predict(X_teste)

        previsoes[teste] = y_pred

        resultados.append(
            {
                "Fold": fold,
                "Acurácia": accuracy_score(y_teste, y_pred),
                "Precisão": precision_score(y_teste, y_pred, zero_division=0),
                "Recall": recall_score(y_teste, y_pred, zero_division=0),
                "F1-score": f1_score(y_teste, y_pred, zero_division=0),
            }
        )

    metricas = pd.DataFrame(resultados).set_index("Fold")

    matriz = confusion_matrix(y, previsoes, labels=[0, 1])

    return metricas, matriz
