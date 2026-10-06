# Relatório — Escada de calibração T0.5 (L0–L9)

**Execução:** 2026-09-06 · seed 2025 · `alpha = 0.05` · `max_lag = 3` · `tau_min = 1` ·
background knowledge ligado · 7 algoritmos × 10 níveis = 70 runs (210 linhas em
`results/runs.csv`, `phase = B`, `spec_version = v2_ladder`).

---

## 1. Objetivo

Os snapshots anteriores (`v1_legacy`, `v2`, `v2_recal`) comparavam algoritmos de descoberta
causal em **um único preset "realista"** (`causal_business`). Nesse desenho, um F1 baixo é
ininterpretável: não se sabe se o método falhou por causa do tamanho amostral, do adstock,
da saturação, do confundidor sazonal ou da colinearidade que o confundidor induz — porque
todos esses fatores estão ligados ao mesmo tempo.

A escada de calibração T0.5 responde a uma pergunta diferente e mais específica:

> **Em qual fator de dificuldade, isoladamente, cada algoritmo colapsa?**

O desenho liga **um fator por degrau**, partindo de um cenário quase-oracular (L0) até o
cenário realista completo (L9). Isso permite três leituras que o preset único não permite:

1. **Separar "o algoritmo é ruim" de "o dado é indetectável".** Se um método falha em L0 —
   série de 20 anos, ruído desprezível, sem confundidor —, o problema é de implementação,
   não de dificuldade do problema.
2. **Atribuir a queda a um fator nomeado.** A diferença de F1 entre L7 e L8 é atribuível
   exclusivamente à concentração de fase sazonal, porque é a única coisa que muda.
3. **Estabelecer um piso de comparabilidade.** Todos os degraus compartilham a mesma
   estrutura causal de base (4 canais → `y` em lag 1, mais 1 spillover `x1 → x2`) e a
   mesma razão sinal/ruído calibrada, de forma que a métrica é comparável entre degraus.

---

## 2. Metodologia — os níveis da escada

### 2.1 Estrutura causal comum a todos os degraus

- 4 canais de mídia `x1..x4`, 1 controle exógeno `c1`, alvo `y`.
- **Arestas verdadeiras de base (5):** `x1→y`, `x2→y`, `x3→y`, `x4→y` (todas em lag 1) e o
  spillover `x1→x2` (lags 1 e 2).
- Quando há sazonalidade, entram no ground truth `season→y` e `season→x_i` para todo canal
  com carga `|loading| > 0.01` — **todas contemporâneas (lag 0)**.
- Os controles **não** entram no ground truth.
- Os betas `canal → y` são recalibrados a cada degrau para manter o sinal total dos canais
  em ~3× o nível de ruído. Sem isso, a razão sinal/ruído variaria junto com os outros
  fatores e a comparação entre degraus seria inválida.

### 2.2 O que cada degrau liga

| Nível | T | Ruído | Fator ligado (acumulativo) | Racional |
|---|---|---|---|---|
| **L0** | 1040 | 0.001 | — (quase-oracular) | Piso de sanidade. 20 anos de dados semanais, ruído desprezível, sem confundidor. Falhar aqui é falha de implementação. |
| **L1** | 1040 | 0.02 | + ruído realista | Isola o efeito do ruído mantendo a série longa. |
| **L2** | 156 | 0.02 | + tamanho amostral realista | 3 anos semanais — o que uma marca real tem. Isola o custo de `T` pequeno. |
| **L3** | 156 | 0.02 | + adstock (`decay ~ U(0.3, 0.5)`) | Carryover geométrico no spend. Autocorrelaciona os regressores. |
| **L4** | 156 | 0.02 | + saturação em `canal → y` | Não linearidade `K·log1p(x/K)` no caminho de efeito. |
| **L5** | 156 | 0.02 | + saturação no spillover | Não linearidade no caminho `x1 → x2`. Testa o teste de independência, não o efeito em `y`. |
| **L6** | 156 | 0.02 | + sazonalidade **observada** (`S1`) | O confundidor existe e está no dado como coluna. Caso favorável: é possível condicionar nele. |
| **L7** | 156 | 0.02 | + sazonalidade **oculta** (`S0`) | Mesmo confundidor, agora latente. Testa robustez a confundimento não observado. |
| **L8** | 156 | 0.02 | + fases concentradas | As cargas sazonais dos 4 canais passam a ter fase `N(0, 0.2)` em vez de equiespaçada: os canais passam a se mover juntos. **Colinearidade induzida pelo confundidor.** |
| **L9** | 156 | 0.02 | tudo junto | Adstock + ambas as saturações + `S0` + fases concentradas. Cenário realista completo. |

A progressão é deliberada em três blocos: **L0→L2** varia só o regime estatístico;
**L3→L5** introduz as não linearidades típicas de MMM; **L6→L8** introduz o confundidor
sazonal em severidade crescente (observável → latente → latente e mal condicionado).

### 2.3 Diagnósticos do dado por degrau

Medidos **antes** de rodar qualquer algoritmo. Nenhuma métrica de desempenho deve ser lida
sem esta tabela ao lado.

| Nível | T | Arestas | Detectáveis | Teto recall | max&#124;r&#124; canais | Nº cond. | Varsort. z | R² médio das arestas |
|---|---|---|---|---|---|---|---|---|
| L0 | 1040 | 5 | 5 | 1.000 | 0.048 | 1.08 | 0.200 | 0.111 |
| L1 | 1040 | 5 | 5 | 1.000 | 0.048 | 1.08 | 0.200 | 0.132 |
| L2 | 156 | 5 | 5 | 1.000 | 0.101 | 1.17 | 0.400 | 0.156 |
| L3 | 156 | 5 | 5 | 1.000 | 0.426 | 1.93 | 0.200 | 0.292 |
| L4 | 156 | 5 | 5 | 1.000 | 0.101 | 1.17 | 0.400 | 0.156 |
| L5 | 156 | 5 | 5 | 1.000 | 0.112 | 1.17 | 0.400 | 0.142 |
| L6 | 156 | 8 | 5 | 0.625 | 0.730 | 2.76 | 0.375 | 0.377 |
| L7 | 156 | 8 | 5 | 0.625 | 0.730 | 2.76 | — | 0.263 |
| L8 | 156 | 10 | 5 | 0.500 | 0.797 | 4.07 | — | 0.716 |
| L9 | 156 | 10 | 5 | 0.500 | 0.911 | 6.86 | — | 0.804 |

Notas de leitura:

- **`max|r|` é a variável que realmente se move:** 0.05 → 0.10 (L2) → 0.73 (L6/L7) →
  0.80 (L8) → 0.91 (L9). Adstock (L3) produz um pico isolado de 0.43 e não persiste.
- **Varsortability z fica `NaN` em L7–L9 por construção** — `season` está no ground truth
  mas não no dado, e o cálculo exige que todas as variáveis da verdade existam. Onde é
  calculável, fica em 0.20–0.40, ou seja, **abaixo do acaso (0.5)**: o benchmark não é
  vencível ordenando variâncias marginais.
- **O teto de recall (0.625 / 0.500) conta as arestas sazonais como indetectáveis** porque
  são contemporâneas (lag 0) e `tau_min = 1`. Ver a ressalva na §5.

### 2.4 Avaliação

Métricas estratificadas por tipo de aresta, comparando a adjacência aprendida com a
verdadeira **na interseção dos espaços de variáveis**:

| Estrato | Pares avaliados | Pergunta |
|---|---|---|
| `overall` | todos os pares `i ≠ j` | desempenho global |
| `channel_to_y` | canais → `y` | os canais efetivos são recuperados? (o estrato que importa para MMM) |
| `channel_to_channel` | todos os pares de canais | o spillover é recuperado sem explodir em FP? |

Em L7–L9 a sazonalidade é latente, então `season` **não está** no grafo aprendido e sai da
submatriz avaliada: nesses degraus a avaliação é sobre as **5 arestas observáveis**, não
sobre as 8/10 do ground truth completo.

---

## 3. Configuração do experimento

| Item | Valor |
|---|---|
| Níveis | L0–L9 (10) |
| Algoritmos | `granger`, `dynotears`, `lpcmci`, `cedar`, `cdnots`, `regime_pcmci`, `rpcmci` (7) |
| **Não executado** | `pcmci_cmiknn` — custo proibitivo (~50 min/run nos presets, ≈ 8 h só para a escada) |
| Seeds | 1 (`seed_data = seed_algo = 2025`) |
| `alpha` / `max_lag` / `tau_min` | 0.05 / 3 / 1 |
| Background knowledge | ligado (13–14 links proibidos: `y → *`, `controle ↔ canal`) |
| Tempo total de parede | ≈ 3.0 h |

---

## 4. Resultados

### 4.1 Tabela principal — F1 no estrato `overall`

Linhas = níveis da escada; colunas = modelos. `—` = F1 indefinido (grafo vazio: sem TP e
sem FP, precisão é `NaN`).

| Nível | granger | dynotears | lpcmci | cedar | cdnots | regime_pcmci | rpcmci |
|---|---|---|---|---|---|---|---|
| **L0** | **1.00** | 0.46 | 0.91 | **1.00** | — | — | — |
| **L1** | **1.00** | 0.46 | 0.91 | **1.00** | — | — | — |
| **L2** | **1.00** | 0.48 | **1.00** | 0.91 | — | — | — |
| **L3** | **1.00** | 0.48 | **1.00** | **1.00** | — | — | — |
| **L4** | **1.00** | 0.48 | **1.00** | 0.91 | — | — | — |
| **L5** | 0.89 | 0.48 | **0.91** | 0.89 | — | — | — |
| **L6** | 0.76 | 0.36 | **0.88** | 0.86 | 0.77 | 0.77 | — |
| **L7** | 0.71 | 0.48 | 0.67 | **0.89** | 0.57 | — | — |
| **L8** | 0.48 | 0.48 | 0.56 | **0.77** | 0.29 | — | — |
| **L9** | 0.48 | 0.50 | 0.59 | **0.73** | — | 0.44 | — |

### 4.2 O mesmo, decomposto em precisão e recall (`overall`)

| Nível | granger | dynotears | lpcmci | cedar | cdnots | regime_pcmci | rpcmci |
|---|---|---|---|---|---|---|---|
| L0 | 1.00 / 1.00 | 0.38 / 0.60 | 0.83 / 1.00 | 1.00 / 1.00 | — / 0.00 | — / 0.00 | — / 0.00 |
| L1 | 1.00 / 1.00 | 0.38 / 0.60 | 0.83 / 1.00 | 1.00 / 1.00 | — / 0.00 | — / 0.00 | — / 0.00 |
| L2 | 1.00 / 1.00 | 0.31 / 1.00 | 1.00 / 1.00 | 0.83 / 1.00 | — / 0.00 | — / 0.00 | — / 0.00 |
| L3 | 1.00 / 1.00 | 0.31 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | — / 0.00 | — / 0.00 | — / 0.00 |
| L4 | 1.00 / 1.00 | 0.31 / 1.00 | 1.00 / 1.00 | 0.83 / 1.00 | — / 0.00 | — / 0.00 | — / 0.00 |
| L5 | 1.00 / 0.80 | 0.31 / 1.00 | 0.83 / 1.00 | 1.00 / 0.80 | — / 0.00 | — / 0.00 | — / 0.00 |
| L6 | 0.62 / 1.00 | 0.25 / 0.62 | 0.88 / 0.88 | 1.00 / 0.75 | 1.00 / 0.62 | 1.00 / 0.62 | — / 0.00 |
| L7 | 0.56 / 1.00 | 0.31 / 1.00 | 0.50 / 1.00 | 1.00 / 0.80 | 1.00 / 0.40 | — / 0.00 | — / 0.00 |
| L8 | 0.31 / 1.00 | 0.31 / 1.00 | 0.38 / 1.00 | 0.62 / 1.00 | 0.50 / 0.20 | — / 0.00 | — / 0.00 |
| L9 | 0.31 / 1.00 | 0.33 / 1.00 | 0.42 / 1.00 | 0.67 / 0.80 | — / 0.00 | 0.50 / 0.40 | — / 0.00 |

*Formato: precisão / recall.*

### 4.3 Estrato `channel_to_y` — recall (precisão é 1.00 em **todas** as células)

| Nível | granger | dynotears | lpcmci | cedar | cdnots | regime_pcmci | rpcmci |
|---|---|---|---|---|---|---|---|
| L0 | 1.00 | 0.75 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| L1 | 1.00 | 0.75 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| L2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| L3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| L4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| L5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| L6 | 1.00 | 1.00 | 1.00 | 0.75 | 0.50 | 0.50 | 0.00 |
| L7 | 1.00 | 1.00 | 1.00 | 0.75 | 0.50 | 0.00 | 0.00 |
| L8 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.00 | 0.00 |
| L9 | 1.00 | 1.00 | 1.00 | 0.75 | 0.00 | 0.50 | 0.00 |

**Zero falsos positivos `canal → y` em 70/70 runs.**

### 4.4 Estrato `channel_to_channel` — falsos positivos (de 12 pares possíveis)

| Nível | granger | dynotears | lpcmci | cedar | cdnots | regime_pcmci | rpcmci |
|---|---|---|---|---|---|---|---|
| L0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 |
| L1 | 0 | 5 | 1 | 0 | 0 | 0 | 0 |
| L2 | 0 | 11 | 0 | 1 | 0 | 0 | 0 |
| L3 | 0 | 11 | 0 | 0 | 0 | 0 | 0 |
| L4 | 0 | 11 | 0 | 1 | 0 | 0 | 0 |
| L5 | 0 | 11 | 1 | 0 | 0 | 0 | 0 |
| L6 | 4 | 11 | 0 | 0 | 0 | 0 | 0 |
| L7 | 4 | 11 | 5 | 0 | 0 | 0 | 0 |
| L8 | **11** | **11** | 8 | 3 | 1 | 0 | 0 |
| L9 | **11** | 10 | 7 | 2 | 0 | 2 | 0 |

Recall do spillover (`x1→x2`) é 1.00 em todos os níveis para dynotears e lpcmci, e para
granger/cedar em todos exceto **L5**, onde ambos vão a 0.00.

### 4.5 Custo computacional (segundos por run)

| Nível | granger | dynotears | lpcmci | cedar | cdnots | regime_pcmci | rpcmci |
|---|---|---|---|---|---|---|---|
| L0 (T=1040) | 0.1 | 0.03 | 0.2 | 596 | 808 | 700 | 405 |
| L1 (T=1040) | 0.1 | 0.03 | 0.3 | 564 | 807 | 674 | 364 |
| L2 | 0.1 | 0.03 | 0.2 | 23 | 285 | 200 | 21 |
| L3 | 0.1 | 0.03 | 0.2 | 24 | 300 | 249 | 17 |
| L4 | 0.1 | 0.03 | 0.2 | 24 | 288 | 212 | 16 |
| L5 | 0.1 | 0.03 | 0.2 | 24 | 219 | 195 | 16 |
| L6 | 0.1 | 0.03 | 0.3 | 41 | 552 | 479 | 31 |
| L7 | 0.1 | 0.03 | 0.3 | 24 | 405 | 342 | 20 |
| L8 | 0.1 | 0.03 | 0.4 | 27 | 503 | 464 | 28 |
| L9 | 0.1 | 0.03 | 0.5 | 26 | 504 | 380 | 23 |
| **Total** | **1.1 s** | **0.3 s** | **2.9 s** | **23 min** | **78 min** | **65 min** | **16 min** |

---

## 5. Achados

**1. Os cinco primeiros degraus não discriminam nada.** `granger`, `lpcmci` e `cedar`
ficam em F1 ≈ 1.00 de L0 a L4. Adstock (L3) e saturação `canal → y` (L4) **não custam
nada** a nenhum método competente — nem mesmo aos lineares, e mesmo com `T = 156`. As duas
não linearidades canônicas de MMM, isoladamente, não são o problema. A escada só começa a
ter poder discriminante a partir de L5.

**2. O colapso é em L8, e a variável causal é colinearidade, não não linearidade.** Entre
L7 e L8 muda **apenas** a dispersão de fase sazonal, e o F1 de granger cai 0.71 → 0.48, o
de lpcmci 0.67 → 0.56, o de cedar 0.89 → 0.77. O que acompanha essa queda é `max|r|`
(0.73 → 0.80 → 0.91) e o número de condição (2.8 → 4.1 → 6.9). Sazonalidade oculta por si
só (L6 → L7) custa pouco; **sazonalidade oculta que faz os canais se moverem juntos**
custa tudo. Esse é o resultado central da escada, e é exatamente o tipo de atribuição que o
preset único não permitia fazer.

**3. O estrato que importa para MMM está resolvido — e o F1 global esconde isso.**
`channel_to_y` tem **precisão 1.00 em todas as 70 células** e recall 1.00 em todos os dez
degraus para granger e lpcmci (e para dynotears de L2 em diante), inclusive em L9.
Nenhum algoritmo, em nenhum
nível, inventou um efeito de canal sobre `y` que não existe. **Todo o erro medido no F1
global vem do estrato `channel_to_channel`.** Reportar apenas o F1 agregado faria parecer
que os métodos "falham em L8/L9" quando, para a pergunta de negócio ("quais canais afetam
vendas?"), eles acertam 4/4 sem nenhum falso positivo.

**4. `cedar` é o mais robusto sob degradação; `granger` é o melhor no fácil.** `cedar`
lidera em L7, L8 e L9 (F1 0.89 / 0.77 / 0.73) e é o único a manter precisão acima de 0.6
nos dois piores degraus. `granger` é ótimo até L4 e o mais frágil a partir de L6 — passa a
emitir 4 a 11 falsos positivos `canal → canal` justamente quando o confundidor aparece,
que é o comportamento esperado de um teste que não condiciona no confundidor.

**5. `rpcmci` devolve grafo vazio em 10/10 níveis — inclusive em L0.** Falhar no degrau
quase-oracular (T = 1040, ruído 0.001, sem confundidor, sem não linearidade) não é
"desempenho baixo": é **ausência de saída**, e localiza o problema na implementação, não na
dificuldade do dado. Isso corrobora o bug já identificado em
`causal_discovery/algorithms/rpcmci.py:91` (leitura de `causal_results["p_matrix"]`, quando
a tigramite devolve um dicionário indexado por regime). Custo do nada: 16 min.

**6. `cdnots` e `regime_pcmci` só produzem saída quando há sazonalidade.** Ambos devolvem
grafo vazio em L0–L5 (todos os degraus sem regime) e só emitem arestas quando o driver
sazonal existe: `cdnots` em L6/L7/L8, `regime_pcmci` em L6 e L9. É o comportamento
coerente de um método baseado em regimes — sem variação de regime, não há o que detectar —
mas significa que **eles são inaplicáveis, não apenas fracos, na maior parte da escada**.
Quando produzem saída, a precisão é alta (1.00 em L6/L7) e o recall é baixo (0.20–0.62):
são conservadores. Custo combinado: 2.4 h, a maior parte para produzir matrizes vazias.

**7. `dynotears` é insensível à escada — o que o desqualifica como sonda de dificuldade.**
Seu F1 fica entre 0.36 e 0.50 do L0 ao L9, sem tendência. Recall 1.00 a partir de L2 com
precisão ~0.31 e 10–11 falsos positivos `canal → canal` em todos os níveis: aceita quase
tudo, sempre. Em L8/L9 ele "empata" com granger (0.48 vs 0.48) não por ter resistido à
degradação, mas porque granger **desceu até o patamar em que dynotears sempre esteve**.
Custo: 0.03 s por run.

**8. L5 isola uma falha específica dos testes paramétricos.** É o único degrau em que
granger e cedar perdem o spillover (recall `channel_to_channel` = 0.00 nos dois), enquanto
lpcmci e dynotears o recuperam. A única mudança em L5 é a saturação `K·log1p(x/K)` **no
caminho `x1 → x2`** — não no caminho para `y`. Ou seja: a não linearidade importa quando
está no caminho que o teste de independência precisa avaliar, e não quando está no efeito
sobre o alvo (L4, onde ninguém perde nada). É uma distinção que só a escada revela.

**9. O teto analítico de recall não é vinculante e foi violado.** Em L6 o teto registrado é
0.625 (as arestas sazonais são contemporâneas, lag 0, e `tau_min = 1`), mas granger obteve
recall 1.00 — 8 de 8 arestas, **incluindo** as sazonais. A razão é que `cos(2πt/52)` em
resolução semanal é quase idêntico ao seu próprio lag 1, então uma aresta contemporânea de
um driver suave é perfeitamente detectável em lag 1. O cálculo de `recall_ceiling` assume
que os lags de uma aresta são exclusivos, o que é falso para drivers autocorrelacionados.
Em L7–L9 o teto de 0.500 está registrado no CSV mas **não se aplica à métrica reportada**:
com `season` latente, ela sai da submatriz avaliada e a avaliação é sobre 5 arestas
observáveis. **Os recalls de 1.00 em L8/L9 se referem a essas 5 arestas, não às 10 do
ground truth.** Corrigir isso exige mudar o cálculo em `runner.py`, não o DGP.

**10. Custo e retorno estão descorrelacionados.** `granger` + `dynotears` + `lpcmci`
custam **4.4 segundos somados para a escada inteira** e colocam pelo menos um deles no
primeiro ou segundo lugar de F1 em todos os dez degraus. Os três métodos caros consomem 2.6 h e produzem
grafo vazio na maioria das células. O único caro que se paga é `cedar` (23 min no total),
que é o melhor método nos três degraus mais difíceis.

---

## 6. Limitações desta execução

1. **Uma única seed (2025).** Não há barra de erro. As diferenças de F1 dentro de ~0.1
   entre métodos vizinhos não são interpretáveis; as afirmações da §5 se apoiam em saltos
   grandes (0.71 → 0.48) e em padrões estruturais (grafo vazio, precisão 1.00 constante),
   que são robustos a ruído de amostragem, mas o achado 2 merece replicação com ≥ 5 seeds.
2. **`pcmci_cmiknn` não foi executado.** É o único método não paramétrico do registry, o
   que deixa a comparação "linear vs. não linear" incompleta justamente nos degraus L4/L5,
   onde a não linearidade é o fator isolado. Custo estimado: ≈ 8 h.
3. **As variantes com canais fantasma não foram rodadas.** `results/escada/diagnosticos_niveis.csv`
   traz os diagnósticos de `L2g` e `L9g` (6 canais, 2 fantasmas), mas não existe nenhuma
   linha correspondente em `runs.csv`. Sem elas, o estrato `ghost_to_y` — o teste direto de
   "o método inventa efeito onde não há?" — não existe na escada.
4. **`control_to_channel` continua impossível.** `run_benchmark` não passa `control_names`
   para `evaluate_stratified`, então o estrato nunca é criado. É mudança de código no
   `runner.py`.
5. **`seed_algo` não tem semântica.** Está registrado no CSV, mas nenhum algoritmo do
   registry aceita semente; a coluna não distingue nada nesta execução.
6. **Os degraus L0/L1 usam T = 1040 e distorcem a comparação de custo.** `cedar` leva 596 s
   em L0 e 23 s em L2 — a diferença é o tamanho da série, não a dificuldade. A leitura de
   custo da §4.5 deve ser feita dentro do bloco L2–L9.

---

## 7. Próximos passos sugeridos

| Prioridade | Ação | Por quê |
|---|---|---|
| Alta | Replicar L6–L9 com 5 seeds | É onde estão todos os achados; é o único bloco onde a seed única realmente pesa |
| Alta | Rodar `L2g` / `L9g` | Fecha o estrato `ghost_to_y`, que é a pergunta mais direta do trabalho |
| Média | Corrigir `recall_ceiling` para lags não exclusivos (achado 9) | O teto atual está sendo violado e não é reportável como está |
| Média | Rodar `pcmci_cmiknn` só em L4, L5 e L9 | 3 runs em vez de 10 fecham a lacuna não paramétrica a ~2.5 h |
| Baixa | Corrigir `rpcmci.py:91` e re-rodar | Hoje o método não contribui nenhuma informação ao benchmark |

---

*Fonte: `new_pipe_dag/results/runs.csv` (`phase = B`), `new_pipe_dag/results/run_ladder_v2_bk.log`,
`new_pipe_dag/results/escada/diagnosticos_niveis.csv`.*
