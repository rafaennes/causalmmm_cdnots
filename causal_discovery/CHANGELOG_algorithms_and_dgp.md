# Descoberta Causal: Expansão de Algoritmos & Reformulação da Geração de Dados

Este documento registra as alterações feitas no módulo `causal_discovery/` em uma única sessão: novos algoritmos informados por uma revisão de literatura, reescrita completa do gerador de dados sintéticos, correções críticas de bugs e resultados finais do benchmark.

---

## 1. Motivação

Uma revisão de literatura sobre descoberta causal para dados de marketing não-estacionários revelou que:

- A implementação existente em `cdnots.py` (detecção de regimes via PELT + PCMCI) era na verdade **Regime-PCMCI** (Família 2 na revisão), e não o verdadeiro método CD-NOD/CD-NOTS com variável substituta (Família 1).
- Dois algoritmos de alto valor estavam ausentes: **CEDAR** (especialista em baixo N) e **RPCMCI** (descoberta conjunta de regimes + estrutura causal, do tigramite).
- O formato de `link_assumptions` no cdnots original estava errado (`"?->"` em vez de `"-?>"`), fazendo o PCMCI **pular todos os testes de IC** e aceitar todas as arestas.

## 2. Alterações nos Algoritmos

### 2.1 Renomeado: `cdnots.py` → `regime_pcmci.py`

O antigo `cdnots.py` detectava regimes via PELT/Binseg (ruptures), codificava-os como uma coluna de contexto inteira discreta, e depois rodava PCMCI nos dados aumentados. É uma abordagem em dois estágios: detectar regimes primeiro, depois descobrir causalidade. Foi renomeado para `regime_pcmci.py` para refletir honestamente o que faz.

### 2.2 Novo: `cdnots.py` — Verdadeiro CD-NOD com Variável Substituta

Implementa o método de Zhang, Huang, Zhang, Glymour & Schölkopf (2017):

- **Fase 1**: Adiciona o índice temporal normalizado contínuo `C = linspace(0, 1, T)` como variável substituta. Roda PCMCI sobre `V ∪ {C}` com `C` restrito como exógeno (nada causa `C`). Testes de IC no conjunto aumentado exploram mudanças distribucionais para identificação.
- **Fase 2**: Identifica **variáveis C-adjacentes** (aquelas cujos mecanismos mudam ao longo do tempo — `C → V_i` é significante). Usa a C-adjacência para orientar arestas: se `V_k` é C-adjacente e `V_l` não é, o triplo não-blindado `C - V_k - V_l` orienta como `V_k → V_l`.

Diferença principal em relação ao `regime_pcmci`: C é contínuo (preserva tendências suaves), não discretizado em inteiros de regime.

### 2.3 Novo: `cedar.py` — Algoritmo CEDAR

Fesanghary (2026). Projetado para dados escassos (T=100-200) com dinâmica de lag-1. O pacote pip `causal-ts` não está disponível no PyPI, então esta é uma implementação manual do algoritmo central:

1. **Residualização AR(1)** por variável para remover correlação serial dominante
2. **Triagem por correlação de distância U-centrada** nos resíduos — O(d²) pares com p-valores baseados em permutação (199 permutações)
3. **Correção BH** nos p-valores da triagem
4. **Poda estilo MCI**: para cada aresta sobrevivente `i→j`, retesta condicionando em todos os outros pais aceitos de `j` via correlação de distância parcial (residualização OLS no conjunto de condicionamento, depois teste dcor)
5. **C-node** opcional: adiciona índice temporal normalizado aos resíduos para ajustar para não-estacionariedade de tendência

### 2.4 Novo: `rpcmci.py` — RPCMCI do Tigramite

Wrapper do `RPCMCI` do tigramite (Saggioro, de Wiljes, Kretschmer & Runge, 2020). Diferente do `regime_pcmci` que detecta regimes primeiro via PELT, o RPCMCI otimiza **conjuntamente** a atribuição de regimes e o grafo causal via alternância MIP (ortools) + PCMCI.

- Usa apenas ParCorr (CMIknn seria proibitivamente lento através de múltiplas iterações de annealing)
- `max_anneal=3`, `num_iterations=10`, `n_jobs=1` para runtime razoável
- Retorna um dict: `result["regimes"]`, `result["causal_results"]`, `result["error_free_annealings"]`
- Necessitou instalar `ortools` como nova dependência

### 2.5 Registro

`algorithms/__init__.py` REGISTRY expandido de 5 para 8 algoritmos:

| Algoritmo | Teste de IC | Trata Não-Estacionariedade | Runtime (T=104, 6 vars) |
|---|---|---|---|
| `granger` | F-test (par a par) | Não | <1s |
| `pcmci_cmiknn` | CMIknn (não-linear) | Não | ~5 min |
| `lpcmci` | ParCorr (linear) | Não (confundidores latentes) | <1s |
| `dynotears` | Score-based (coefs VAR) | Não | <1s |
| `cdnots` | CMIknn + substituta C | Sim (variável substituta) | ~3 min |
| `regime_pcmci` | CMIknn + regime discreto | Sim (detecção de regime) | ~3 min |
| `cedar` | Correlação de distância | Parcial (C-node) | ~10s |
| `rpcmci` | ParCorr + regimes conjuntos | Sim (otimização conjunta) | ~8s |

## 3. Correções de Bugs

### 3.1 Formato de `link_assumptions` (crítico)

O `cdnots.py` original (e agora tanto `cdnots.py` quanto `regime_pcmci.py`) usava:

```python
la[j][(i, lag)] = "?->"   # ERRADO
```

O tigramite interpreta `"?->"` como **"assuma que este link existe, apenas oriente-o"** — pulando todos os testes de independência condicional. O formato correto é:

```python
la[j][(i, lag)] = "-?>"   # CORRETO: teste este link
```

**Impacto**: Com `"?->"`, o PCMCI aceitava todas as arestas (30/30 em um sistema de 6 variáveis) em 0.0s. Com `"-?>"`, ele executa os testes CMIknn corretamente (~200s) e retorna um grafo seletivo. Este bug estava presente nos 4 commits originais na main — todo resultado do cdnots anterior a esta correção era inválido.

### 3.2 NaN do CMIknn com variável substituta determinística

O teste de significância por block bootstrap do CMIknn calcula um comprimento ótimo de bloco usando a autocorrelação `phi` de cada variável:

```
l_opt = (4T * (phi/(1-phi) + phi²/(1-phi)²)² / (1 + 2*phi/(1-phi))²)^(1/3)
```

O índice temporal determinístico `C = linspace(0, 1, T)` tem `phi = 1.0`, causando divisão por zero → NaN → `ValueError: cannot convert float NaN to integer`.

**Correção**: Definir `sig_blocklength=1` no construtor do CMIknn para forçar permutações simples (sem blocos):

```python
ci_test = CMIknn(knn=5, sig_samples=500, sig_blocklength=1)
```

Isso se aplica tanto ao `cdnots.py` (C contínuo) quanto ao `regime_pcmci.py` (inteiro de regime discreto com phi≈1 dentro dos regimes).

### 3.3 Tipo de retorno do RPCMCI

O `run_rpcmci()` do RPCMCI retorna um **dict** (`{"regimes": ..., "causal_results": ..., ...}`), não uma tupla. A implementação inicial descompactava como tupla, causando `AttributeError: 'str' object has no attribute 'get'`. Corrigido para `result = rpcmci.run_rpcmci(...)` e depois `result["regimes"]`, etc. Também trata o caso de retorno `None` (todos os annealings falharam).

## 4. Reformulação da Geração de Dados Sintéticos

### 4.1 Problemas com o DGP original

O `_synthetic_preset()` original em `compare.py` tinha três problemas críticos:

1. **Sinal canal→y indetectável**: Usava saturação `sqrt(spend)` com spend na casa dos milhares (sqrt(2000) ≈ 45) e coeficientes de efetividade 0.5-1.5, produzindo ~45 por canal por timestep contra desvio padrão de ruído de 200. Correlação bivariada bruta canal→y era **r ≈ 0** (p > 0.5).

2. **Efeitos apenas em lag-1**: Tanto as arestas canal→y quanto inter-canal eram apenas em lag 1. A dissertação requer lag-1 E lag-2.

3. **Timing de regime compartilhado**: Todos os canais tinham sua mudança de regime em `T//2`. Isso criava correlações cruzadas espúrias massivas (r > 0.7) entre TODOS os pares de canais, tornando arestas inter-canal verdadeiras indistinguíveis das falsas.

### 4.2 Tentativa intermediária: VAR(2)

Um modelo VAR(2) foi tentado como substituto: `V(t) = A1 @ V(t-1) + A2 @ V(t-2) + eps(t)` com a estrutura causal codificada nas matrizes de coeficientes. Isso produziu separação de sinal limpa mas **não era dado de marketing realista** — faltava adstock, saturação, sazonalidade e escalas de investimento realistas. Era um benchmark de algoritmos de livro-texto, não uma simulação de MMM.

### 4.3 DGP final: Realista para marketing com sinais controlados

A implementação final combina realismo de marketing com qualidade de sinal:

```
Passo 1: Inovações de investimento semanal independentes por canal
  - Cada canal tem seu próprio nível base ($500-2000/semana)
  - Mudança de regime por canal em tempo aleatório em [T/4, 3T/4]
  - Direção da mudança aleatória (para cima ou para baixo), magnitude 30-60%
  - Ruído semana-a-semana: 15% do nível base
  - Piso: mínimo $50/semana

Passo 2: Efeitos causais inter-canal (no nível da inovação)
  - Aplicados ANTES do adstock para que o sinal seja limpo
  - Spillover em lag-1: 15-25% do investimento da fonte
  - Spillover em lag-2: 5-12% do investimento da fonte

Passo 3: Adstock geométrico (carryover)
  - Taxa de decaimento 0.3-0.6 por canal (decaimento semanal realista)
  - Aplicado após a injeção causal

Passo 4: Saturação (retornos decrescentes)
  - log(1 + x/K) onde K = mediana do investimento por canal
  - Cria relação não-linear canal→y

Passo 5: Geração de vendas (y)
  - Vendas base: $8k-15k/semana
  - Sazonalidade trimestral: onda senoidal com amplitude de 3%
  - Efeitos dos canais em lag 1 E lag 2 através do investimento saturado
  - SNR por canal calibrado: std total do sinal ≈ 2× std do ruído
  - Ruído: std de $500/semana

Passo 6: Controles
  - Variáveis aleatórias N(0,1) com efeito fraco (±$50)
```

### 4.4 Verificação de sinal

**Preset causal_business** (4 canais, T=104, aresta verdadeira: x1→x2):

| Par | r bruto lag-1 | p Granger | Aresta verdadeira? |
|---|---|---|---|
| x1→y | 0.576 | 0.007 | Sim |
| x2→y | 0.697 | 0.033 | Sim |
| x3→y | 0.596 | 0.007 | Sim |
| x4→y | 0.504 | 0.041 | Sim |
| x1→x2 | 0.459 | 0.044 | **Sim** |
| x3→x1 | 0.769 | 0.000 | Não (espúria) |

Propriedades-chave:
- **Todas as arestas verdadeiras canal→y detectáveis** no nível bivariado bruto (r > 0.4, p < 0.05)
- **Aresta inter-canal verdadeira (x1→x2) claramente direcional**: r=0.459 na direção correta vs r=0.049 na reversa
- **Correlações espúrias existem** (x3→x1 r=0.77) devido à suavização por adstock — isso é realista e cria um desafio significativo para os algoritmos
- **Saturação log cria não-linearidade**: ParCorr (teste de IC linear) perde algumas arestas canal→y que CMIknn (não-linear) deveria detectar
- **Não-estacionariedade cria arestas espúrias**: x3→x1 sobrevive até ao condicionamento do PCMCI — exatamente o fenômeno de Zhang et al. que o CD-NOTS foi projetado para corrigir

## 5. Atualização dos Testes

`test_algorithms.py` expandido de 6 para 9 testes:

| Teste | Algoritmo | Asserções principais |
|---|---|---|
| `test_granger` | granger | formato da adjacência, restrição de y |
| `test_pcmci_cmiknn` | pcmci_cmiknn | pvalues nos metadados |
| `test_lpcmci` | lpcmci | graph_matrix nos metadados |
| `test_dynotears` | dynotears | w_threshold, n_edges_raw nos metadados |
| `test_cdnots` | cdnots | c_adjacent_vars, n_c_adjacent nos metadados |
| `test_regime_pcmci` | regime_pcmci | regime_boundaries, n_regimes nos metadados |
| `test_regime_pcmci_fixed_regimes` | regime_pcmci | n_regimes == 2 quando forçado |
| `test_cedar` | cedar | dcor_matrix, use_c_node nos metadados |
| `test_rpcmci` | rpcmci | regimes, num_regimes nos metadados |

Todos os 17 testes passam (9 algoritmos + 6 graph + 2 compare). Runtime total ~8-9 minutos, dominado pelos testes baseados em CMIknn.

## 6. Resultados do Benchmark

Comparação completa no `causal_business` (5 arestas verdadeiras, T=104):

| Algoritmo | Precisão | Recall | F1 | FP | FN | Tempo |
|---|---|---|---|---|---|---|
| **granger** | **0.75** | **0.60** | **0.67** | 1 | 2 | 0.1s |
| pcmci_cmiknn | 0.67 | 0.40 | 0.50 | 1 | 3 | 316s |
| lpcmci | 0.50 | 0.40 | 0.44 | 2 | 3 | 0.4s |
| dynotears | 0.31 | 1.00 | 0.48 | 11 | 0 | 0.2s |
| cdnots | 0.40 | 0.40 | 0.40 | 3 | 3 | 193s |
| regime_pcmci | 0.25 | 0.20 | 0.22 | 3 | 4 | 205s |
| cedar | 0.00 | 0.00 | 0.00 | 0 | 5 | 12s |
| rpcmci | 0.00 | 0.00 | 0.00 | 0 | 5 | 8s |

### Interpretação

1. **Granger lidera em F1 (0.67)** — A correção BH nos testes par a par é eficaz nesta escala. Encontra 3/5 arestas verdadeiras com apenas 1 falso positivo.

2. **PCMCI+CMIknn (0.50)** — Melhor precisão que DYNOTEARS mas menor recall. O teste de IC não-linear ajuda, mas condicionar em muitas variáveis com T=104 reduz o poder estatístico.

3. **DYNOTEARS tem recall perfeito (1.0) mas precisão terrível (0.31)** — o limiar de coeficientes VAR é muito permissivo com dados adstockeados. Todo par tem coeficientes cross-lag não-triviais.

4. **CD-NOTS (0.40)** — A variável substituta identifica 4 variáveis C-adjacentes (x1, x2, x4, y) e usa isso para orientação. Porém, produz 3 falsos positivos — a variável substituta ajuda na orientação mas não elimina todas as arestas espúrias induzidas por não-estacionariedade com T=104.

5. **Regime-PCMCI (0.22)** — Pior que CD-NOTS. Discretizar o tempo em regimes perde informação comparado à substituta contínua, e os limites de regime podem não se alinhar com a estrutura real de não-estacionariedade.

6. **CEDAR e RPCMCI (0.00)** — Não encontram nada. A triagem por correlação de distância do CEDAR após residualização AR(1) é muito agressiva — remove o sinal causal junto com a autocorrelação. O RPCMCI usa ParCorr (linear) e com T=104 dividido entre 2 regimes, não tem poder estatístico suficiente.

### O que isso significa para a dissertação

- T=104 com dados de marketing realistas (adstock + saturação + não-estacionariedade) é **genuinamente difícil** para todos os algoritmos
- A não-linearidade da saturação prejudica testes de IC lineares (métodos baseados em ParCorr)
- O adstock cria correlações cruzadas espúrias que desafiam todas as abordagens
- Nenhum algoritmo domina — a **abordagem de ensemble** sugerida na revisão de literatura (consenso entre múltiplos algoritmos → força do prior) é bem motivada por estes resultados
- CEDAR e RPCMCI precisam de ajuste ou podem não ser adequados para este tamanho de amostra

## 7. Arquivos Alterados

| Arquivo | Alteração |
|---|---|
| `algorithms/cdnots.py` | Reescrito: método com variável substituta |
| `algorithms/regime_pcmci.py` | Novo: código antigo do cdnots movido para cá |
| `algorithms/cedar.py` | Novo: algoritmo de correlação de distância |
| `algorithms/rpcmci.py` | Novo: wrapper do RPCMCI do tigramite |
| `algorithms/__init__.py` | Registry: 5 → 8 algoritmos |
| `compare.py` | `_synthetic_preset()` reescrito |
| `tests/test_algorithms.py` | 6 → 9 testes |

### Dependências adicionadas

- `ortools` (para o solver MIP do RPCMCI)
