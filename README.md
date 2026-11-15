# Grupo 9 — Online Shoppers Purchasing Intention

## Execução

Com Python 3.10 ou superior, na pasta do projeto:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook
```

Execute primeiro `script/01_analise_exploratoria.ipynb` a partir da pasta `script` para atualizar o CSV convertido. Depois execute todas as células de `script/02_knn.ipynb` em ordem. O segundo notebook aceita execução a partir da raiz do projeto ou da pasta `script` e grava tabelas e figuras em `resultados/`. Selecione no Jupyter o kernel do ambiente em que as dependências foram instaladas.

## Protocolo compartilhado

`src/preprocessamento.py` define o conjunto de atributos, `carregar_dados`, `preparar_dados`, `criar_preprocessador` e `criar_particoes`. Os demais modelos devem usar o mesmo CSV convertido, as mesmas funções e os mesmos dados preparados, na mesma ordem, e inserir um novo preprocessador em cada pipeline. As partições são estratificadas, embaralhadas e fixadas pela semente 42; a imputação, a padronização e a codificação são ajustadas apenas no treino de cada partição.

`src/avaliacao.py` fornece `avaliar_modelo(modelo, X, y, particoes)`. O parâmetro `modelo` deve ser um `Pipeline` com `criar_preprocessador()` e o classificador. A função clona e treina o pipeline em cada fold, retorna um DataFrame de métricas por fold e uma matriz de confusão agregada com previsões fora da amostra. Ela verifica se cada registro é validado exatamente uma vez e se não há sobreposição entre treino e validação de cada fold. Exemplo de uso com duas configurações está no notebook de KNN.