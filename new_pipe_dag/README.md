# `new_pipe_dag` — Harness de benchmark para descoberta causal em dados de MMM

Documento descritivo do que está **proposto e implementado** nesta pasta: arquitetura,
DGP paramétrico ("escada de calibração" T0.5), avaliação estratificada, diagnósticos,
esquema de saída, resultados já materializados em `results/` e as lacunas conhecidas.

> Nomenclatura: o código referencia uma "spec v2" externa (tarefas `T0.1`, `T0.2`, `T0.5`,
> `T0.7`, anexos `A0.3`, `A2`, `A5` e o esquema `§9`). Esse documento de especificação **não
> está versionado no repositório**; o que segue foi reconstruído lendo o código e os
> artefatos gerados.

---

## 1. Objetivo

O módulo `causal_discovery/` já comparava algoritmos de descoberta causal em um preset
sintético e devolvia um número agregado de F1/SHD. O `new_pipe_dag` é uma reescrita do
laço de benchmark com três propostas centrais:

1. **Diagnosticar o dado antes de culpar o algoritmo.** Antes de rodar qualquer método,
   mede-se colinearidade, varsortability, fração de spend zero, força de sinal das arestas
   verdadeiras (R² parcial) e um teto analítico de recall. Se o dado é indetectável, F1
   baixo não é evidência sobre o algoritmo.
2. **Métricas estratificadas por tipo de aresta.** Um F1 global mistura arestas
   `canal → y`, `canal → canal` e falsos positivos em canais *ghost* (efetividade zero).
   Cada estrato responde a uma pergunta diferente do trabalho e é reportado separadamente.
3. **Escada de calibração (`T0.5`, níveis L0–L9).** Um DGP paramétrico próprio em que cada
   degrau liga **um** fator de dificuldade por vez (ruído, tamanho amostral, adstock,
   saturação, sazonalidade observada/oculta, concentração de fase). Serve para localizar
   em qual fator o desempenho colapsa, em vez de observar apenas o resultado final no
   preset "realista".

Além disso, todo run persiste os dados gerados e o ground truth em disco, e escreve uma
linha por `seed × algoritmo × estrato` num CSV de esquema fixo.

---

## 2. Estrutura da pasta

```
new_pipe_dag/
├── __init__.py        # docstring do pacote
├── __main__.py        # CLI: python -m new_pipe_dag ...
├── dgp.py             # DGP paramétrico + escada T0.5 (L0..L9)
├── evaluation.py      # métricas estratificadas, varsortability, colinearidade
├── runner.py          # loaders, diagnósticos, laço de benchmark, esquema §9
├── experimento_escada.ipynb   # experimento completo: escada, fantasmas, figuras, evidências
└── results/
    ├── escada/                # saída do notebook (runs.csv, figuras/, data/)
    ├── runs.csv                 # run mais recente (sobrescrito a cada execução)
    ├── v1_legacy/runs.csv       # snapshot: DGP v1_legacy, 5 algoritmos
    ├── v2/runs.csv              # snapshot: DGP v2
    ├── v2_recal/runs.csv        # snapshot: DGP v2 (repetição)
    └── data/<preset>_seed<N>_<spec>/
        ├── data.csv             # dados gerados
        ├── true_adj.csv         # matriz de adjacência verdadeira
        ├── variable_names.csv   # ordem das variáveis do ground truth
        └── ghost_names.csv      # canais com efetividade zero
```

---

## 3. `dgp.py` — DGP paramétrico e a escada T0.5

### 3.1 `LadderConfig`

Dataclass com um botão por fator de dificuldade:

| Campo | Default | O que controla |
|---|---|---|
| `n_channels` | 4 | número de canais de mídia |
| `n_controls` | 1 | número de controles exógenos `c{j}` |
| `n_periods` | 1040 | comprimento da série (semanas) |
| `inter_channel_edges` | `[(0, 1)]` | arestas de spillover `(src_idx, tgt_idx)` |
| `noise_std` | 0.001 | razão ruído/sinal em `y` |
| `adstock_on` | False | carryover geométrico no spend |
| `saturation_cy` | False | saturação log no caminho `canal → y` |
| `saturation_spill` | False | saturação log no caminho de spillover |
| `seasonality` | `"off"` | `"off"`, `"S1"` (sazonalidade observada), `"S0"` (oculta) |
| `phase_spread` | `"equispaced"` | dispersão de fase das cargas sazonais: `equispaced`, `moderate`, `concentrated` |
| `seed` | 2025 | semente |

### 3.2 Os dez degraus (`LADDER`)

| Nível | T | ruído | Fator isolado que é ligado |
|---|---|---|---|
| **L0** | 1040 | 0.001 | Caso quase-oracular: série longa, ruído desprezível. Piso de sanidade — se um algoritmo falha aqui, o problema é o algoritmo. |
| **L1** | 1040 | 0.02 | + ruído realista (série ainda longa) |
| **L2** | 156 | 0.02 | + tamanho amostral realista (3 anos semanais) |
| **L3** | 156 | 0.02 | + adstock (carryover geométrico, decay ∈ [0.3, 0.5]) |
| **L4** | 156 | 0.02 | + saturação em `canal → y` |
| **L5** | 156 | 0.02 | + saturação no caminho de spillover |
| **L6** | 156 | 0.02 | + sazonalidade **observada** (`S1`: `season` entra como coluna do dado) |
| **L7** | 156 | 0.02 | + sazonalidade **oculta** (`S0`: confundidor latente, não observado) |
| **L8** | 156 | 0.02 | + fases concentradas (canais correlacionados pela sazonalidade comum) |
| **L9** | 156 | 0.02 | Tudo junto: adstock + ambas as saturações + `S0` + fases concentradas |

A progressão é deliberada: L0→L2 varia só o regime estatístico; L3→L5 introduz as não
linearidades típicas de MMM; L6→L8 introduz o confundidor sazonal, primeiro observável,
depois latente, depois latente e mal condicionado; L9 é o cenário "realista" completo.

### 3.3 `generate_ladder(cfg, seed)` — geração passo a passo

1. **Driver sazonal.** `season_signal = cos(2π t / 52)` (ciclo anual em resolução semanal).
   As cargas por canal são `0.3 · cos(φ_i)`, com as fases `φ` conforme `phase_spread`:
   `equispaced` → `linspace(0, 2π)`; `moderate` → `N(0, 1)`; `concentrated` → `N(0, 0.2)`
   (fases próximas ⇒ canais movendo juntos ⇒ colinearidade induzida pelo confundidor).
2. **Inovações dos canais.** Cada canal recebe base `U(800, 2000)`, ruído
   `N(0, 0.12 · base)` e, se ativa, a parcela sazonal; clip inferior em 50.
3. **Spillover inter-canal.** Para cada aresta `(src, tgt)`, coeficiente `U(0.15, 0.25)`,
   aplicado em **lag 1** e, com peso `0.4×`, em **lag 2**. Com `saturation_spill`, o sinal
   da fonte passa antes por `K · log1p(x / K)` com `K = mediana(x)`.
4. **Adstock.** Se ligado, `x[t] += decay · x[t-1]` com `decay ~ U(0.3, 0.5)`.
5. **Canais → `y` em lag 1.** Os betas são **calibrados** para que o sinal total dos canais
   fique em torno de 3× o nível de ruído: `noise_level = 500 · noise_std / 0.02` (com piso
   0.5 para L0) e `beta_i = (3 · noise_level / n_ch) / std(s_i) · U(0.7, 1.0)`. Isso mantém
   a relação sinal/ruído comparável entre degraus, em vez de deixá-la variar junto com os
   outros fatores. Com `saturation_cy`, `s_i` é saturado antes tanto da calibração quanto
   da soma em `y`.
6. **Baseline, sazonalidade e controles em `y`.** `base_sales = 10000`; se sazonalidade
   ativa, `y += 0.05 · base_sales · season_signal`; cada controle entra com coeficiente
   `U(-30, 30)`; por fim ruído `N(0, noise_level)`.
7. **DataFrame.** Colunas `x1..xN`, `c1..cM`, `season` (apenas em `S1`), `y`.

### 3.4 Ground truth

O espaço da verdade é `canais + [season] + y` — note que **`season` entra no ground truth
mesmo quando é latente (`S0`)**, e os **controles não entram**. As arestas são:

- todo canal → `y`;
- cada `(src, tgt)` de `inter_channel_edges`;
- se sazonalidade ativa: `season → y` e `season → canal_i` para todo canal com carga
  `|loading| > 0.01`.

---

## 4. `evaluation.py` — avaliação estratificada e diagnósticos

### 4.1 `evaluate_stratified(...)`

Compara adjacência aprendida vs. verdadeira restringindo a comparação a subconjuntos de
pares `(i, j)`:

| Estrato | Pares avaliados | Pergunta que responde |
|---|---|---|
| `overall` | todos os pares `i ≠ j` | desempenho global |
| `channel_to_y` | canais não-ghost → `y` | os canais efetivos são recuperados? |
| `ghost_to_y` | canais ghost → `y` | o método inventa efeito onde não há? (só falsos positivos são possíveis) |
| `channel_to_channel` | todos os pares de canais | spillover é recuperado sem explodir em FP? |
| `control_to_channel` | controles → canais | previsto, mas **nunca criado** — `run_benchmark` não passa `control_names` (ver lacuna #3) |

Cada estrato devolve `tp, fp, fn, tn, precision, recall, f1, fdr, shd`. As métricas são
`NaN` quando o denominador é zero — é o caso sistemático de `ghost_to_y`, onde não existem
positivos verdadeiros: **esse estrato deve ser lido pela contagem de `fp`, não por P/R/F1.**

### 4.2 `varsortability(adj, data)`

Métrica de Reisach, Seiler & Weichwald (2021): fração de arestas dirigidas em que
`var(fonte) < var(alvo)`. 0.5 = acaso; próximo de 1.0 = o grafo é recuperável apenas
ordenando variâncias marginais, o que torna o benchmark artificialmente fácil. Calculada
duas vezes: nos dados brutos (`varsortability_raw`) e após padronização
(`varsortability_z`), justamente para separar "sinal causal" de "artefato de escala".

### 4.3 `collinearity_diagnostics(data, names)`

`max|r|`, `mediana |r|` (triângulo superior da matriz de correlação dos canais), número de
condição e VIF (`mean_vif`, `max_vif`; `VIF_j = 1/(1 - R²_j)` regredindo cada canal nos
demais), todos calculados **sobre os dados padronizados** — número de condição em dados
brutos mediria diferença de unidade, não colinearidade. O VIF captura colinearidade
multivariada que o `|r|` par-a-par não vê.

---

## 5. `runner.py` — orquestração

### 5.1 Dois carregadores com a mesma assinatura

Ambos retornam `(data, var_names, channel_cols, control_cols, ghost_names, true_adj, truth_var_names)`:

- **`load_preset(preset_name, seed, spec_version)`** — usa `cdnots.presets.get_preset_config`
  + `cdnots.core.generate_mmm_dataset`. Detecta canais por regex `^x\d+` (excluindo colunas
  de contribuição) e controles por `^c\d+$`; se houver coluna `geo`, mantém apenas o
  primeiro geo. **Ghosts são identificados automaticamente** como canais com
  `base_effectiveness == 0.0`. Se `cdnots` não importar, cai para o gerador sintético de
  `causal_discovery.compare._load_preset`.
- **`load_ladder(level_name, seed)`** — usa `LADDER` + `generate_ladder` de `dgp.py`.

O flag `spec_version` (`v1_legacy` | `v2`) é repassado ao `cdnots`, que mantém as duas
versões do DGP lado a lado: `v1_legacy` preserva bugs originais (spend com drift negativo,
período sazonal errado, zeros estruturais) para comparabilidade com a qualificação, e `v2`
os corrige. O harness torna essa escolha explícita e a registra em cada linha do CSV.

### 5.2 Diagnósticos calculados por dataset (antes dos algoritmos)

- `collinearity_diagnostics` sobre as colunas de canal;
- `varsortability` bruta e padronizada — **só é calculada quando todas as variáveis do
  ground truth existem no dado**; com sazonalidade oculta (`S0`) o resultado é `NaN` por
  construção;
- `recall_ceiling(...)` (`T0.1`) — teto analítico de recall dado `max_lag` e `tau_min`;
- `edge_r2(...)` (`T0.7`) — para cada aresta verdadeira, ganho de R² de uma regressão
  lag-1 `alvo ~ fonte_{t-1}` contra o modelo só-intercepto; reporta a média. É o proxy
  direto de "há sinal detectável nessa aresta?";
- `pct_zero_spend` (`A0.3`) — fração de células de spend exatamente zero.

### 5.3 Laço principal

Para cada `(tarefa, seed)`: carrega → **persiste dados e ground truth** em
`results/data/<nome>_seed<N>_<spec>/` → calcula diagnósticos → para cada algoritmo do
`REGISTRY` de `causal_discovery.algorithms` (`granger`, `bvar_horseshoe`, `pcmci_cmiknn`,
`lpcmci`, `dynotears`, `cdnots`, `regime_pcmci`, `cedar`, `rpcmci`), cronometra a chamada
`fn(data, var_names, alpha=..., max_lag=...)` e avalia.

**Alinhamento verdade × aprendido.** O grafo aprendido e o ground truth vivem em espaços de
variáveis diferentes (o ground truth não tem controles; pode ter nós latentes como
`season`). O runner calcula `observable_truth = truth_vars ∩ learned_names` e extrai a
submatriz correspondente dos dois lados (`_extract_submatrix`), avaliando apenas na
interseção. Falhas de um algoritmo individual são capturadas e reportadas sem abortar o
grid; falhas no carregamento pulam a tarefa inteira.

### 5.4 Esquema de saída (`§9`) — 39 colunas, uma linha por `seed × algoritmo × estrato`

| Grupo | Colunas |
|---|---|
| Identificação | `run_id`, `timestamp`, `phase` (`B` para escada, `exploratory` para preset) |
| Dado | `preset`, `task_type`, `spec_version`, `seed_data`, `seed_algo` |
| Algoritmo | `algorithm`, `ci_test`, `alpha`, `max_lag`, `tau_min` |
| Métricas | `stratum`, `tp`, `fp`, `fn`, `tn`, `precision`, `recall`, `f1`, `fdr`, `shd` |
| Diagnósticos | `recall_ceiling`, `max_abs_corr_channels`, `median_abs_corr_channels`, `condition_number`, `mean_vif_channels`, `max_vif_channels`, `varsortability_raw`, `varsortability_z`, `pct_zero_spend`, `mean_r2_partial_edges` |
| Dimensões | `n_channels`, `n_controls`, `n_ghosts`, `n_periods`, `n_true_edges`, `runtime_seconds` |

O ponto do esquema é que **cada métrica vem acompanhada, na mesma linha, das condições em
que foi obtida** — um F1 nunca é lido sem `max_abs_corr`, `varsortability_z` e
`mean_r2_partial_edges` ao lado.

---

## 6. `__main__.py` — CLI

```bash
python -m new_pipe_dag \
  --presets causal_business \
  --ladder L0 L1 L2 \
  --algorithms granger pcmci_cmiknn lpcmci \
  --seeds 2025 \
  --alpha 0.05 \
  --max-lag 3 \
  --spec-version v2 \
  --output-dir new_pipe_dag/results
```

| Flag | Default | Nota |
|---|---|---|
| `--presets` | — | nomes de preset do `cdnots` (`causal_business`, `causal_large`) |
| `--ladder` | — | níveis `L0`–`L9`, ou `all` (expande para os dez) |
| `--algorithms` | todos do `REGISTRY` | |
| `--seeds` | `[2025]` | aceita múltiplas |
| `--alpha` | 0.05 | nível dos testes de independência |
| `--max-lag` | 3 | |
| `--spec-version` | `v1_legacy` | aplica-se apenas a presets, não à escada |
| `--output-dir` | `new_pipe_dag/results` | |

Ao final imprime `groupby(preset, algorithm, stratum)` com média de P/R/F1/SHD.

---

## 7. Resultados já materializados

Todos os snapshots existentes são do preset **`causal_business`**, seed 2025:
8 canais (6 efetivos + 2 ghosts: `x7_Ghost-A`, `x8_Ghost-B`), 1 controle, T = 156,
9 arestas verdadeiras (6 `canal → y` + 3 `canal → canal`), `alpha = 0.05`, `max_lag = 3`.

### 7.1 Diagnósticos do dado

| Snapshot | max&#124;r&#124; | mediana&#124;r&#124; | cond. | varsort. bruta | varsort. z | % spend zero | R² médio das arestas |
|---|---|---|---|---|---|---|---|
| `runs.csv` (fallback) | 0.867 | 0.138 | 4.80 | 0.778 | 0.667 | 24.3% | 0.295 |
| `v1_legacy` | 0.867 | 0.106 | 4.44 | 0.778 | 0.667 | 24.6% | 0.295 |
| `v2` / `v2_recal` | 0.669 | 0.101 | 3.15 | 0.667 | 0.444 | 6.7% | 0.161 |

Leitura: o DGP `v2` é **materialmente diferente e mais honesto** — reduz colinearidade
máxima (0.87 → 0.67), quase elimina os zeros estruturais de spend (24.6% → 6.7%) e derruba
a varsortability padronizada de 0.667 para 0.444, ou seja, aproxima-a do acaso (0.5) e
remove o atalho de "ordenar por variância". O preço é sinal por aresta mais fraco
(R² médio 0.295 → 0.161): o problema fica genuinamente mais difícil.

### 7.2 Métricas por algoritmo (estrato `overall`)

**`v1_legacy`** (`results/v1_legacy/runs.csv`)

| Algoritmo | TP | FP | FN | P | R | F1 | SHD | runtime |
|---|---|---|---|---|---|---|---|---|
| `cdnots` | 2 | 1 | 7 | 0.667 | 0.222 | **0.333** | 8 | 1944 s |
| `lpcmci` | 1 | 5 | 8 | 0.167 | 0.111 | 0.133 | 13 | 3.3 s |
| `pcmci_cmiknn` | 1 | 9 | 8 | 0.100 | 0.111 | 0.105 | 17 | 3216 s |
| `regime_pcmci` | 0 | 0 | 9 | 0.000 | 0.000 | 0.000 | 9 | 1946 s |
| `rpcmci` | 0 | 0 | 9 | 0.000 | 0.000 | 0.000 | 9 | 88 s |

**`v2`** (`results/v2/runs.csv`; `v2_recal` é idêntico em métricas, difere só em runtime)

| Algoritmo | TP | FP | FN | P | R | F1 | SHD | runtime |
|---|---|---|---|---|---|---|---|---|
| `lpcmci` | 4 | 3 | 5 | 0.571 | 0.444 | **0.500** | 8 | 3.5 s |
| `cdnots` | 2 | 1 | 7 | 0.667 | 0.222 | 0.333 | 8 | 1166 s |
| `pcmci_cmiknn` | 1 | 11 | 8 | 0.083 | 0.111 | 0.095 | 19 | 2963 s |
| `regime_pcmci` | 0 | 0 | 9 | 0.000 | 0.000 | 0.000 | 9 | 940 s |
| `rpcmci` | 0 | 0 | 9 | 0.000 | 0.000 | 0.000 | 9 | 24 s |

**Run inicial** (`results/runs.csv`, `spec_version` gravado como `fallback`)

| Algoritmo | TP | FP | FN | P | R | F1 | SHD | runtime |
|---|---|---|---|---|---|---|---|---|
| `dynotears` | 9 | 47 | 0 | 0.161 | 1.000 | 0.277 | 47 | 0.02 s |
| `lpcmci` | 1 | 2 | 8 | 0.333 | 0.111 | 0.167 | 10 | 5.3 s |
| `granger` | 1 | 7 | 8 | 0.125 | 0.111 | 0.118 | 15 | 0.6 s |

### 7.3 Onde os erros acontecem — o valor da estratificação

| Snapshot / algoritmo | `channel_to_y` | `channel_to_channel` | `ghost_to_y` (FP) |
|---|---|---|---|
| v1 `cdnots` | 1/6 recuperadas, 0 FP | 1/3, 1 FP | 0 |
| v1 `pcmci_cmiknn` | **0/6**, 0 FP | 1/3, 9 FP | 0 |
| v1 `lpcmci` | **0/6**, 0 FP | 1/3, 4 FP | 1 |
| v2 `lpcmci` | 2/6, 0 FP | 2/3, 3 FP | 0 |
| v2 `pcmci_cmiknn` | **0/6**, 0 FP | 1/3, 11 FP | 0 |
| v2 `cdnots` | 1/6, 0 FP | 1/3, 1 FP | 0 |
| fallback `dynotears` | **6/6, 0 FP** | 3/3 mas **45 FP** | 2 |
| fallback `granger` | 0/6 | 1/3, 7 FP | 0 |

Três padrões que o F1 agregado esconderia:

- Os métodos baseados em PCMCI **quase não recuperam `canal → y`** (o estrato que mais
  importa para MMM), e quase todos os seus falsos positivos estão em `canal → canal`.
- `dynotears` tem recall perfeito com precisão desprezível: acerta 9/9 arestas e adiciona
  47 falsos positivos — o comportamento de um método que aceita quase tudo, invisível se
  apenas o recall fosse reportado.
- `regime_pcmci` e `rpcmci` devolvem **grafo vazio** nas condições testadas (0 TP, 0 FP),
  consumindo 16–32 minutos para isso. Não é "baixa performance": é ausência de saída.
- Custo computacional varia por 5 ordens de grandeza: `dynotears` 0.02 s vs.
  `pcmci_cmiknn` ~53 min.

---

## 8. Lacunas e limitações conhecidas

Legenda de estado: **[corrigida]** já resolvida no código; **[notebook]** endereçada pelo
`experimento_escada.ipynb`; **[aberta]** permanece.

1. **[corrigida] A escada T0.5 nunca havia sido executada.** Hoje `results/escada/runs.csv`
   tem 920 linhas com `phase = B` (L0–L9 + variantes fantasma `L2g`/`L8g`, 5 seeds),
   produzidas pelo notebook. Os snapshots antigos (`v1_legacy`, `v2`, `v2_recal`) seguem só
   com `task_type = preset`.
2. **[corrigida] `recall_ceiling` era um stub** que retornava `1.0` incondicionalmente.
   Hoje `runner.py` recebe `edge_lags` de `dgp.py` e de `CausalEdgeConfig` e conta as
   arestas cujo lag cai em `[tau_min, max_lag]`. Efeito imediato: na escada as arestas
   sazonais são contemporâneas (lag 0), então o teto vale 0.625 em L6/L7 e 0.500 em L8/L9.
3. **[aberta] `control_to_channel` é impossível, não apenas ausente.** O estrato não deixa
   de existir por falta de controles no ground truth: `run_benchmark` (`runner.py:307-312`)
   **não passa `control_names`** para `evaluate_stratified`, e o default `()` deixa a lista
   vazia em toda tarefa. Corrigir exige mudança no `runner.py`, não no DGP.
4. **[notebook] Uma única seed (2025) em todos os snapshots.** Sem replicação não há barra
   de erro; as diferenças entre `v1_legacy` e `v2` na tabela 7.2 (por exemplo `lpcmci`
   0.133 → 0.500) são de uma amostra só. O notebook usa 5 seeds nos blocos A/B/D e 3 nos
   blocos C/E.
5. **[aberta] `seed_algo` não altera nada.** O parâmetro foi separado de `seed_data` e é
   registrado no CSV, mas nenhum algoritmo do `REGISTRY` aceita semente — `run_benchmark`
   chama `fn(data, var_names, alpha=, max_lag=)` e nada mais. Pior: os laços de `seeds` e
   `seed_algos` são aninhados (`runner.py:223` e `:278`) e `seed_algos` tem como padrão o
   próprio `seeds`, então passar 5 seeds numa só chamada executa cada algoritmo 25 vezes,
   produzindo linhas duplicadas. O notebook contorna passando sempre listas de um elemento.
6. **[notebook] `v2` e `v2_recal` são runs duplicados** — métricas idênticas, apenas
   `run_id`, `timestamp` e `runtime_seconds` diferem. Não são replicações independentes; o
   bloco E as substitui por 3 seeds reais por versão do DGP.
7. **[corrigida] `results/runs.csv` era sobrescrito a cada execução.** Hoje o
   `run_benchmark` grava em modo *append*; o notebook acrescenta a retomada por checkpoint
   e uma leitura com deduplicação defensiva.
8. **[notebook] `granger` e `dynotears` só aparecem no run `fallback`**, e `cedar` não havia
   sido executado; as tabelas `v1_legacy`/`v2` cobrem apenas 5 dos 9 algoritmos do
   `REGISTRY`. A escada (`results/escada/`) já roda `granger`, `dynotears`, `lpcmci`,
   `cedar` e `rpcmci` em todos os níveis; `cdnots`, `regime_pcmci` e `bvar_horseshoe` têm
   cobertura parcial, e `pcmci_cmiknn` não aparece na escada.
9. **[aberta, por construção] Varsortability é `NaN` sempre que houver nó latente** no
   ground truth (caso `S0`, níveis L7–L9), pelo guard em `runner.py:247-254`. A Fig. 3 do
   notebook torna esses buracos visíveis em vez de escondê-los.
10. **[notebook, com correção de descrição] `ghost_to_y` se comporta de dois modos.** No
    preset o estrato **existe** com P/R/F1 `NaN` (não há positivos verdadeiros possíveis);
    na escada base ele **não existe** — `evaluate_stratified` faz `continue` quando o
    conjunto de pares é vazio, então nem a linha é gravada. Em ambos os casos a leitura
    correta é pela contagem de `fp`. O bloco D do notebook cria o estrato na escada.

## 9. Estado atual e próximos passos

O experimento que fecha as lacunas acima está em
[`experimento_escada.ipynb`](experimento_escada.ipynb) — escada L0–L9, variantes com canais
fantasma, matriz completa no preset, e as seções de evidência das falhas de `rpcmci` e
`regime_pcmci`. Consulte a §16 do notebook para as instruções de execução (o `.venv` exige
`LD_LIBRARY_PATH`; os blocos longos rodam em tmux).

Permanecem em aberto, exigindo mudança de código:

- Passar `control_names` em `run_benchmark` para que o estrato `control_to_channel` exista.
- Dar semântica real ao `seed_algo`: algum algoritmo precisa aceitar semente, e os laços
  aninhados de `seeds`/`seed_algos` precisam virar um `zip` ou um produto explícito.

Corrigidos no código (commit `d604e8a`), mantidos aqui como registro das evidências nas
§12–§13 do notebook:

- **[corrigida]** `rpcmci.py` lia `causal_results["p_matrix"]` como dicionário plano, mas a
  tigramite devolve `{regime_k: resultados_pcmci}` — o método sempre retornava grafo vazio.
  Hoje agrega os resultados por regime.
- **[corrigida]** `regime_pcmci` aplicava BH sobre ~147–300 testes com o CMIknn tendo piso de
  valor-p em 1/501, o que impedia qualquer aresta de sobreviver. Hoje usa `pc_alpha=None` e
  valores-p brutos, como o `pcmci_cmiknn`.
