# Trabalho final da disciplina de IC

## Visão Geral do Projeto

Este projeto é o Trabalho Final da disciplina de Inteligência Computacional e tem como objetivo o desenvolvimento e a análise de um sistema inteligente para a classificação de cinco personagens da série *Os Simpsons* (Homer, Marge, Bart, Lisa e Maggie).

O foco principal do trabalho foi comparar a eficácia de diferentes estratégias de extração de características e demonstrar o impacto da fusão (Ensemble) de classificadores no desempenho final.

## Tecnologias e Métodos

O sistema é construído em Python e utiliza as seguintes ferramentas e técnicas:

| Categoria | Componentes Implementados |
| :--- | :--- |
| **Extração de Características** | HOG, Histograma de Cores (HSV) e **Deep Features (VGG16)**. |
| **Classificadores Base** | k-NN, Árvore de Decisão, SVM, MLP e Random Forest. |
| **Estratégia Final** | **Ensemble Estático (Soft Voting)** com 20 modelos diversos. |
| **Validação** | Cross-Validation Estratificada de 10 Folds. |
| **Bibliotecas** | `Scikit-Learn`, `TensorFlow/Keras` (VGG16), `OpenCV`, `joblib`. |

## Resultado Principal

O modelo final de **Ensemble** (utilizando as Deep Features da VGG16) atingiu a acurácia máxima de **74.63%** no conjunto de teste fixo, superando o desempenho de todos os classificadores individuais.

## Como Executar

1.  **Treinamento e Avaliação:**
    * O script `main.py` executa todo o fluxo de trabalho (extração de features, CV, treino, teste fixo e Ensemble).
    * Gera as matrizes de confusão e o relatório `resultados/relatorio_desempenho.md`.
    * **Importante:** Ele também salva o modelo Ensemble e o Scaler na pasta `resultados/` nos arquivos `.pkl` para uso em tempo real.

2.  **Teste do treinamento:**
    * Utilize o script `web_app` para carregar os resultados salvos no passo anterior e abrir a aplicação web onde é possível fazer o upload de uma imagem e o sistema tentara identificar o personagem que estiver na imagem.

## Estrutura de Arquivos

* `main.py`: O pipeline principal de treinamento e avaliação.
* `tools/`: Contém as funções de `data_loader`, `feature_extractor`, `classifier_model` e `results_handler`.
* `resultados/`: Pasta criada durante a execução do main.py, irá conter o `relatorio_desempenho.md`, as matrizes de confusão (`.png`) e os modelos salvos (`simpsons_ensemble_model.pkl` e `simpsons_scaler.pkl`).
* `data/`: Pasta de entrada com as subpastas dos personagens (dataset).

