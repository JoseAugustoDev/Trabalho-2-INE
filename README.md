# Grupo 9 — Online Shoppers Purchasing Intention

## Instalação no Windows

O ambiente foi testado em Windows 64 bits com **Python 3.14**. As versões das bibliotecas principais estão fixadas em `requirements.txt`, para reduzir diferenças entre computadores. Versões futuras do Python podem exigir outras versões dessas bibliotecas e não foram testadas.

Abra o **PowerShell na pasta raiz do projeto**, onde está este README, e execute:

```powershell
py -3.14 --version
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name trabalho2-ine --display-name "Python (.venv Trabalho 2)"
```

Se o comando `py` não existir, confirme que `python --version` informa Python 3.14 e use `python -m venv .venv` na segunda linha. A instalação precisa de acesso à internet. Não é necessário ativar o ambiente nem alterar a política de execução do PowerShell: os comandos usam diretamente o Python do projeto.

O último comando registra um kernel **dentro do próprio `.venv`**, sem alterar os kernels globais do usuário. Crie um ambiente `.venv` novo **em cada computador**; não copie a pasta de outro computador. Caso já exista um ambiente do projeto com outra versão de Python, renomeie ou exclua essa pasta somente depois de conferir seu conteúdo e, então, recrie-a com os comandos acima.

## Abrir os notebooks

**No VS Code:** abra a pasta inteira do projeto, instale as extensões Python e Jupyter e abra um notebook. No canto superior direito, clique em **Selecionar Kernel** → **Ambientes Python** → escolha `.venv\Scripts\python.exe` deste projeto. Se a lista não mostrar o ambiente, use a opção para informar o caminho do interpretador. Execute as células em ordem. Repita a seleção se outro notebook abrir com um kernel diferente.

**No navegador:** a partir da raiz do projeto, execute:

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Na interface aberta, entre na pasta `script`, abra o notebook desejado e escolha **Kernel → Change Kernel → Python (.venv Trabalho 2)**. Mesmo iniciando o Jupyter pelo ambiente do projeto, um notebook pode estar configurado para usar outro kernel; selecione explicitamente o kernel criado acima.

Em ambos os casos, uma célula com `import sys; print(sys.executable)` deve mostrar o caminho de `.venv\Scripts\python.exe`. Se aparecer outro Python, selecione o kernel correto e **reinicie o kernel** antes de executar as células novamente. Se um pacote estiver ausente, refaça a instalação usando o comando com `.venv\Scripts\python.exe` acima; instalar em outro terminal ou interpretador não instala automaticamente no kernel do notebook.

## Ordem de execução

Execute os notebooks a partir da raiz do projeto ou da pasta `script`:

1. `script/01_analise_exploratoria.ipynb`: lê a base original e gera `dados/online_shoppers_convertido.csv`.
2. `script/02_knn.ipynb`.
3. `script/03_svm.ipynb`.
4. `script/04_random_forest.ipynb`.
5. `script/05_xgboost.ipynb`.

Execute todas as células de cada notebook em ordem. O primeiro precisa ser executado antes dos modelos caso o CSV convertido ainda não exista ou tenha sido atualizado. As tabelas e figuras são gravadas na pasta `resultados`, criada automaticamente quando necessário. Executar novamente pode sobrescrever arquivos de resultados com os mesmos nomes.

## Protocolo compartilhado

`src/preprocessamento.py` define o conjunto de atributos, `carregar_dados`, `preparar_dados`, `criar_preprocessador` e `criar_particoes`. Os demais modelos devem usar o mesmo CSV convertido, as mesmas funções e os mesmos dados preparados, na mesma ordem, e inserir um novo preprocessador em cada pipeline. As partições são estratificadas, embaralhadas e fixadas pela semente 42; a imputação, a padronização e a codificação são ajustadas apenas no treino de cada partição.

`src/avaliacao.py` fornece `avaliar_modelo(modelo, X, y, particoes)`. O parâmetro `modelo` deve ser um `Pipeline` com `criar_preprocessador()` e o classificador. A função clona e treina o pipeline em cada fold, retorna um DataFrame de métricas por fold e uma matriz de confusão agregada com previsões fora da amostra. Ela verifica se cada registro é validado exatamente uma vez e se não há sobreposição entre treino e validação de cada fold. Exemplo de uso com duas configurações está no notebook de KNN.
