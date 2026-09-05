# Pipeline Granger Estacionário

## Visão Geral

Este pipeline testa se estacionarizar os dados antes de aplicar causalidade de Granger melhora a recuperação da estrutura causal nos presets sintéticos de MMM.

## Fluxo do Pipeline

```
1. Carregar Dataset
   │  _load_preset(preset_name) de compare.py
   │  Retorna: DataFrame bruto, nomes das variáveis, matriz de adjacência verdadeira
   │
2. Aplicar Transformação de Estacionaridade
   │  Para cada variável:
   │    a. Executar teste ADF (Augmented Dickey-Fuller, autolag="AIC")
   │    b. Se p-valor >= 0.05 → aplicar primeira diferenciação, re-testar
   │    c. Se ainda não estacionária → aplicar segunda diferenciação
   │    d. Parar em d=2 no máximo
   │  Alinhar todas as séries ao menor comprimento após diferenciação
   │
3. Verificar Estacionaridade
   │  Re-executar ADF em cada variável estacionarizada
   │  Imprimir WARNING para qualquer variável que ainda falhe (p >= 0.05)
   │  Reportar taxa de aprovação: "ADF check: N/N stationary"
   │
4. Executar Causalidade de Granger
   │  Testes Granger par-a-par (statsmodels) nos dados estacionarizados
   │  Correção FDR de Benjamini-Hochberg na matriz de p-valores
   │  Restrição: y não causa nada (arestas y→* zeradas)
   │  Saída: matriz de adjacência aprendida (CausalGraph)
   │
5. Avaliar Contra Ground Truth
   │  Comparar grafo aprendido vs adjacência verdadeira (canais + y apenas)
   │  Métricas: Precision, Recall, F1, FDR, SHD, TP, FP, FN, TN
   │
6. Armazenar Resultados
      Por preset → causal_discovery/results/stationary/<preset>/
        - adjacency_matrix.csv       (grafo aprendido)
        - true_adjacency_matrix.csv  (ground truth)
        - diff_orders.csv            (ordem de diferenciação por variável)
        - adf_pvals_post.csv         (p-valores ADF após estacionarização)
      Combinado → causal_discovery/results/stationary/
        - metrics.csv                (todos os presets × todas as métricas)
```

## Uso

```bash
# Executar em todos os presets
.venv/bin/python3.12 -m causal_discovery.stationary.run_stationary_granger

# Executar em presets específicos
.venv/bin/python3.12 -m causal_discovery.stationary.run_stationary_granger --presets causal_business causal_large

# Parâmetros customizados
.venv/bin/python3.12 -m causal_discovery.stationary.run_stationary_granger --alpha 0.10 --max-lag 3
```

## Parâmetros

| Parâmetro   | Padrão  | Descrição                                          |
|-------------|---------|---------------------------------------------------|
| `--presets` | todos   | Quais presets sintéticos executar                   |
| `--alpha`   | 0.05    | Nível de significância para ADF e Granger           |
| `--max-lag` | 2       | Lag máximo para os testes de causalidade de Granger  |

## Arquivos de Saída

| Arquivo                    | Conteúdo                                                      |
|----------------------------|---------------------------------------------------------------|
| `adjacency_matrix.csv`     | Adjacência binária aprendida (linhas=causa, colunas=efeito)   |
| `true_adjacency_matrix.csv`| Adjacência verdadeira do DGP                                  |
| `diff_orders.csv`          | Quantas vezes cada variável foi diferenciada (0, 1 ou 2)      |
| `adf_pvals_post.csv`       | P-valores ADF após estacionarização (todos devem ser < 0.05)  |
| `metrics.csv`              | Precision, Recall, F1, FDR, SHD por preset                   |
