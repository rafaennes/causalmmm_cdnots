# Reescrita dos Capítulos de Metodologia e Implementação — Plano de Execução

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reescrever as seções 6 (Metodologia) e 7 (Implementação) da dissertação como dois capítulos coesos, usando o experimento de 4 braços como fio condutor, com trechos de código Python real anotados e justificativas formais para a banca acadêmica.

**Architecture:** O Capítulo 6 é acadêmico/formal (sem código), cobrindo design experimental, justificativas algorítmicas e derivações matemáticas. O Capítulo 7 é o espelho técnico do Capítulo 6, com trechos de código real de cada módulo. As seções são numeradas para correspondência 1:1 (6.2↔7.2, 6.3↔7.3, etc.).

**Tech Stack:** Markdown (ABNT adaptado para dissertação Mackenzie), trechos Python 3.10+, tigramite/PCMCI, PyMC-Marketing, Meridian/TFP.

**Output file:** `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` — substituir seções 6, 7 e 8 pelos novos capítulos 6 e 7.

**Reference files (read-only):**
- `arquivos_recentes/cdnots_discovery.py` — código de descoberta
- `arquivos_recentes/cdnots_model_builder.py` — código de calibração
- `arquivos_recentes/cdnots_fitter.py` — código de fitting
- `arquivos_recentes/presets.py` — preset causal_business
- `arquivos_recentes/config.py` — dataclasses CausalEdgeConfig, MMMDataConfig

---

## Verificação Global (executar ao final de cada capítulo)

1. Cap. 6 legível sem código — banca sem Python deve entender a metodologia
2. Cada fórmula em 6.4 tem correspondente de código em 7.4
3. MIN_SIGMA_RATIO = 0,4, BASE_B = 3,0, MIN_B = 1,0 aparecem consistentemente
4. Trechos de código batem com arquivos reais (não inventados)
5. Nenhuma seção tem placeholder "[INSERIR...]" no final

---

## Task 1: Substituir seções 6, 7, 8 — scaffold dos dois novos capítulos

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md`

- [ ] **Passo 1: Localizar as seções a substituir**

No arquivo, identificar onde começa `# 6. Solução Proposta` e onde termina `# 8. Validação` (inclusive). Tudo entre esses marcadores será substituído pelos dois novos capítulos.

- [ ] **Passo 2: Inserir scaffold vazio dos dois capítulos**

Substituir o bloco inteiro de seções 6, 7, 8 por:

```markdown
# 6. Metodologia

## 6.1 Visão Geral: O Experimento de 4 Braços

[RASCUNHO — ver Task 2]

## 6.2 Dados Sintéticos com Estrutura Causal Conhecida

[RASCUNHO — ver Task 3]

## 6.3 Descoberta da Estrutura Causal via CD-NOTS

[RASCUNHO — ver Task 4]

## 6.4 Tradução do Grafo em Priors Estruturais (Empirical Bayes)

[RASCUNHO — ver Task 5]

## 6.5 Frameworks Bayesianos e o Framework de Comparação

[RASCUNHO — ver Task 6]

## 6.6 Métricas de Avaliação

[RASCUNHO — ver Task 7]

---

# 7. Implementação

## 7.1 Arquitetura do Sistema

[RASCUNHO — ver Task 8]

## 7.2 Geração de Dados: `config.py` e `presets.py`

[RASCUNHO — ver Task 9]

## 7.3 Módulo de Descoberta Causal: `cdnots_discovery.py`

[RASCUNHO — ver Task 10]

## 7.4 Módulo de Calibração de Priors: `cdnots_model_builder.py`

[RASCUNHO — ver Task 11]

## 7.5 Ajuste dos Modelos: `cdnots_fitter.py`

[RASCUNHO — ver Task 12]

## 7.6 Framework de Benchmark e Notebook

[RASCUNHO — ver Task 13]

## 7.7 Reprodutibilidade

[RASCUNHO — ver Task 14]
```

- [ ] **Passo 3: Verificar estrutura**

Confirmar que as seções 9, 10, 11, 12, 13, 14 (Resultados, Impactos etc.) continuam intactas após o novo scaffold.

- [ ] **Passo 4: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "chore: scaffold dois novos capítulos Metodologia e Implementação"
```

---

## Task 2: Escrever Seção 6.1 — Visão Geral: O Experimento de 4 Braços

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 6.1)

- [ ] **Passo 1: Escrever parágrafo de abertura com as três questões de pesquisa**

```markdown
## 6.1 Visão Geral: O Experimento de 4 Braços

Esta pesquisa avalia empiricamente a hipótese de que grafos causais descobertos algoritmicamente podem melhorar a calibração de modelos Bayesianos de MMM em contextos de dados escassos. O experimento é estruturado para responder a três questões específicas:

1. O algoritmo CD-NOTS consegue descobrir estruturas causais informativas entre canais de marketing a partir de séries temporais com 104–208 observações por geo?
2. A tradução do grafo causal descoberto em ajustes de priors Bayesianos melhora as métricas preditivas e de atribuição dos modelos PyMC-Marketing e Meridian?
3. Qual o ganho marginal da descoberta causal sobre os modelos baseline, e em quais cenários esse ganho é mais pronunciado?
```

- [ ] **Passo 2: Inserir tabela dos 4 braços**

```markdown
O design experimental de 4 braços foi construído para isolar o efeito da descoberta causal mantendo constante o framework Bayesiano utilizado:

| Braço | Descoberta Causal | Framework | Priors |
|-------|-------------------|-----------|--------|
| 1 | Nenhuma (manual) | PyMC-Marketing | Padrão (spend-share) |
| 2 | Nenhuma (manual) | Meridian | Padrão (spend-share) |
| 3 | CD-NOTS | PyMC-Marketing | Calibrados pelo grafo |
| 4 | CD-NOTS | Meridian | Calibrados pelo grafo |

A comparação entre os braços 1 vs. 3 (PyMC) e 2 vs. 4 (Meridian) isola o efeito da descoberta causal. Se os braços com CD-NOTS superarem consistentemente seus respectivos baselines, demonstra-se que a informação causal estrutural agrega valor independentemente do framework Bayesiano utilizado.
```

- [ ] **Passo 3: Inserir diagrama ASCII do pipeline ponta-a-ponta**

```markdown
O pipeline de execução segue o fluxo abaixo, onde todos os braços recebem exatamente os mesmos dados:

```
Dataset Sintético causal_business (156 semanas × 4 geos × 8 canais)
         │
         ├─────────────────────────────────────────┐
         │  (todos os braços usam dados idênticos)  │
         ▼                                         ▼
   [Braços 1 e 2]                          [Braços 3 e 4]
   Priors padrão                            CD-NOTS Discovery
   (proporcional ao spend)                         │
         │                             ┌───────────┘
         │                             ▼
         │                   Tradução: Grafo → Priors
         │                   (Empirical Bayes, Seção 6.4)
         │                             │
         │               ┌────────────┴────────────┐
         │               ▼                         ▼
         │       PyMC + CD-NOTS           Meridian + CD-NOTS
         │               │                         │
         └───────────────┴─────────────────────────┘
                                  │
                           Avaliação Unificada
              SHD / Precision / ROAS / R² / R-hat (Seção 6.6)
```

Os Braços 1 e 2 são executados via o framework de benchmark `mmm_param_recovery` sem modificação. Os Braços 3 e 4 estendem esse framework com os módulos de descoberta e calibração desenvolvidos nesta pesquisa, descritos em detalhe na Seção 6.3 e 6.4, respectivamente.
```

- [ ] **Passo 4: Verificar e fazer commit**

Confirmar que a seção 6.1 não contém `[RASCUNHO]` e que o diagrama está correto.

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 6.1 — visão geral do experimento de 4 braços"
```

---

## Task 3: Escrever Seção 6.2 — Dados Sintéticos com Estrutura Causal Conhecida

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 6.2)
- Reference: `arquivos_recentes/config.py` (CausalEdgeConfig), `arquivos_recentes/presets.py` (causal_business)

- [ ] **Passo 1: Escrever subseção 6.2.1 — Motivação**

```markdown
### 6.2.1 Motivação para Dados Sintéticos

A avaliação do pipeline proposto requer dados com estrutura causal **conhecida a priori** — condição impossível de satisfazer com dados reais de marketing, onde o grafo causal verdadeiro é precisamente o que se busca descobrir. A utilização de dados sintéticos com ground truth explícito permite calcular métricas de recuperação estrutural (SHD, Precision, Recall, FDR) com referência exata, e avaliar se os canais com efeito causal real são distinguidos dos canais sem efeito (ghost channels) pelo pipeline proposto.

Adicionalmente, dados sintéticos eliminam restrições de confidencialidade, permitem controle experimental preciso — variando a estrutura causal sem alterar outros parâmetros — e garantem reprodutibilidade completa do experimento.
```

- [ ] **Passo 2: Escrever subseção 6.2.2 — Framework de configuração**

```markdown
### 6.2.2 Framework de Configuração: `MMMDataConfig`

A geração de dados é parametrizada via o dataclass `MMMDataConfig`, que especifica completamente a estrutura causal e os parâmetros de geração. Os campos principais são:

- `n_periods`: horizonte temporal em semanas
- `channels: List[ChannelConfig]` — nome, padrão de gasto (`linear_trend`, `seasonal`, `on_off`), efetividade base
- `causal_edges: List[CausalEdgeConfig]` — relações causais inter-canal que compõem o ground truth
- `regions: RegionConfig` — número de geos, vendas base, tendência, sazonalidade
- `transforms: TransformConfig` — tipo de adstock e função de saturação
- `control_variables: List[ControlConfig]` — variáveis de controle exógenas (e.g., preço)

O campo `causal_edges` é central para esta pesquisa: ele especifica as relações inter-canal que o algoritmo CD-NOTS deve descobrir, permitindo avaliar quantas arestas verdadeiras foram identificadas e quantas arestas falsas foram introduzidas.
```

- [ ] **Passo 3: Escrever subseção 6.2.3 — Preset causal_business**

```markdown
### 6.2.3 Preset `causal_business`: Estrutura Causal Conhecida

O preset `causal_business` foi desenvolvido especificamente para o benchmark CD-NOTS, simulando uma empresa com presença em 4 regiões geográficas e um funnel de marketing realista com relações causais inter-canal explícitas. Suas dimensões são:

| Dimensão | Valor |
|----------|-------|
| Períodos | 156 semanas (3 anos) |
| Geos | 4 (geo_a, geo_b, geo_c, geo_d) |
| Canais com efeito real | 6 (Search-Ads, Brand-Search, TV, Video, Social-Media, Display-Ads) |
| Canais ghost | 2 (Ghost-A, Ghost-B: efetividade = 0) |
| Controles | 1 (preço, efeito = −0,3) |
| Arestas causais inter-canal | 3 |

Os **canais ghost** (Ghost-A e Ghost-B) têm `base_effectiveness=0.0` — nenhum efeito real nas vendas — mas mantêm padrões de gasto estatisticamente plausíveis (sazonalidade, on-off). Sua inclusão testa a capacidade do pipeline de suprimir canais irrelevantes: o prior calibrado deve apresentar multiplicador próximo de `MIN_SIGMA_RATIO = 0,4` para esses canais.

As **três arestas causais inter-canal** modelam o funnel upper-funnel → lower-funnel:

| Aresta | Defasagem | Tamanho do efeito | Interpretação mercadológica |
|--------|-----------|-------------------|-----------------------------|
| TV → Search-Ads | 2 semanas | 0,20 | Publicidade de branding em TV aumenta a busca paga 2 semanas depois |
| Social-Media → Brand-Search | 1 semana | 0,15 | Engajamento social impulsiona busca por marca na semana seguinte |
| Video → Social-Media | 1 semana | 0,10 | Vídeo online aumenta engajamento em redes sociais |

A **endogeneidade** é introduzida via preço → Search-Ads: reduções de preço (promoções) aumentam o volume de busca paga, confundindo a estimativa de efetividade do canal. O CD-NOTS deve detectar essa relação e o módulo de calibração deve penalizar o prior de Search-Ads proporcionalmente ao R² da confundência.
```

- [ ] **Passo 4: Escrever subseção 6.2.4 — Mecanismo de spillover causal**

```markdown
### 6.2.4 Mecanismo de Spillover Causal

A geração de dados opera em duas fases sequenciais para cada geo:

**Fase 1 — Spend base independente:** cada canal gera sua série temporal de investimento conforme seu padrão configurado, com variações regionais determinísticas. Os canais são independentes entre si nesta fase.

**Fase 2 — Spillover causal:** para cada aresta `CausalEdgeConfig`, o spend do canal fonte é adstockado geometricamente e então adicionado ao spend do canal alvo com a defasagem configurada. As equações de geração são:

$$\text{adstocked}[t] = \text{source\_spend}[t] + \text{decay} \times \text{adstocked}[t-1]$$
$$\text{spillover}[t] = \text{effect\_size} \times \text{adstocked}[t - \text{lag}], \quad t \geq \text{lag}$$
$$\text{channel\_spends}[\text{target}][t] \mathrel{+}= \text{spillover}[t]$$

O parâmetro `effect_size` representa a fração do spend adstockado do canal fonte que se manifesta como spend adicional no canal alvo. As arestas são processadas em ordem de definição; efeitos mediados (Video → Social-Media → Brand-Search) emergem composicionalmente ao longo dos períodos.

### 6.2.5 Ground Truth para Avaliação

A função `_build_causal_ground_truth()` constrói a matriz de adjacência verdadeira a partir da configuração do preset, registrando: (a) arestas inter-canal definidas em `causal_edges`; (b) arestas canal → y para canais com `base_effectiveness > 0`; (c) ausência de arestas para canais ghost. Esta matriz é a referência para calcular as métricas estruturais descritas na Seção 6.6.
```

- [ ] **Passo 5: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 6.2 — dados sintéticos com estrutura causal conhecida"
```

---

## Task 4: Escrever Seção 6.3 — Descoberta da Estrutura Causal via CD-NOTS

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 6.3)

- [ ] **Passo 1: Escrever subseção 6.3.1 — Por que CD-NOTS?**

```markdown
## 6.3 Descoberta da Estrutura Causal via CD-NOTS

### 6.3.1 Justificativa da Escolha Algorítmica

A seleção do algoritmo de descoberta causal foi guiada por três critérios derivados das características dos dados de MMM: (1) operacionalidade com amostras pequenas (N = 104–208 observações por geo); (2) capacidade de capturar não-estacionariedade temporal, característica inerente de séries de marketing; e (3) controle formal da taxa de falsas descobertas, necessário quando o espaço de hipóteses é grande.

A tabela abaixo compara os principais métodos considerados:

| Método | Tipo | Limitação para MMM com low-N |
|--------|------|------------------------------|
| Granger pairwise | Constraint-based | Não controla confounders; ignora mediação inter-canal |
| VAR (Vector Autoregression) | Paramétrico | Requer estacionariedade; sem controle de FDR |
| DYNOTEARS (Pamfil et al., 2020) | Score-based | Score-based requer N >> p; sem controle de FDR |
| CausalMMM / Graph VAE (Gong et al., 2024) | Deep learning | N ≥ 100 entidades geográficas; não quantifica incerteza |
| LPCMCI (Gerhardus & Runge, 2020) | Constraint-based | Modela confounders latentes, mas menos eficiente computacionalmente |
| **CD-NOTS + PCMCI** | **Constraint-based** | **Não-paramétrico; low-N; FDR controlado; não-estacionário ✓** |

O CD-NOTS (Sadeghi, Gopal & Fesanghary, 2024) estende o CD-NOD (Huang et al., 2020) para séries temporais ao adicionar um nó temporal indexado U_t que captura não-estacionariedade sem requerer janelas deslizantes ou segmentação manual do período. Utiliza o PCMCI (Runge, 2020) como motor de descoberta, que resolve o problema de explosão combinatória do PC clássico por meio do teste MCI (Momentary Conditional Independence): ao condicionar nos pais causais de ambas as variáveis testadas simultaneamente, os conjuntos de condicionamento permanecem pequenos independentemente do número de variáveis.

Conforme a recomendação empírica de Sadeghi et al. (2024), ParCorr (correlação parcial) é preferível para N < 200 observações — compatível com o regime típico de dados de MMM semanal analisados nesta pesquisa.
```

- [ ] **Passo 2: Escrever subseções 6.3.2 e 6.3.3**

```markdown
### 6.3.2 Os Quatro Estágios do CD-NOTS

O algoritmo opera em quatro estágios sequenciais:

1. **Adição do nó temporal U_t:** um nó auxiliar indexado pelo tempo é adicionado ao grafo, atuando como causa comum de todas as variáveis não-estacionárias. Isso permite ao algoritmo capturar mudanças de regime sem requerer testes de estacionariedade explícitos.

2. **Descoberta de esqueleto causal:** para cada par de variáveis (X_i, X_j) e cada defasagem τ ∈ {1, ..., τ_max}, executa-se o teste MCI com `tau_min=1` (relações contemporâneas são excluídas por serem ambíguas em dados de marketing com granularidade semanal).

3. **Orientação de arestas:** arestas são orientadas pela regra de precedência temporal (X_{t-τ} → Y_t implica causalidade direcional) e por V-structures (padrões de colisão no grafo).

4. **Orientação residual:** arestas não orientadas pelos estágios anteriores são orientadas pelo critério de independência de mudanças causais, explorando a hipótese de que causas e efeitos variam independentemente sob intervenções externas.

### 6.3.3 Seleção Adaptativa do Teste de Independência Condicional

A implementação usa seleção adaptativa do teste CI em função do número de geos disponíveis:

**Multi-geo (≥ 2 geos) → ParCorr** (correlação parcial com estatística Fisher-Z):
- 100–1000× mais rápido que métodos kernel (KCIT, RCoT)
- A replicação de padrões entre múltiplos geos compensa parcialmente a premissa de linearidade
- Efeitos de adstock e saturação são aproximadamente lineares nos segmentos observacionais de dados semanais com N ≥ 100 por geo

**Single-geo (1 geo) → CMIknn** (estimador k-NN de informação mútua condicional):
- Não-paramétrico; captura não-linearidades características de curvas de adstock e saturação
- Custo computacional O(N³) é aceitável com N ≤ 200 observações
- Requer biblioteca `numba` para execução eficiente
```

- [ ] **Passo 3: Escrever subseções 6.3.4 a 6.3.6**

```markdown
### 6.3.4 Controle de Múltiplos Testes: Correção Benjamini-Hochberg

O PCMCI realiza testes de independência condicional para cada par (i, j) de variáveis e cada defasagem τ ∈ {1, ..., τ_max}. Com 10 variáveis (8 canais + 1 controle + y) e τ_max = 2, o número de testes simultâneos é da ordem de 10 × 10 × 2 = 200. A um nível de significância α = 0,05, esperam-se aproximadamente 10 falsos positivos esperados sob a hipótese nula global — número inaceitável para um grafo com esparsidade esperada de 5–15 arestas verdadeiras.

Aplica-se a correção de Benjamini-Hochberg (BH; 1995), que controla a taxa de falsas descobertas FDR = E[FP/(FP+TP)] ao nível nominal α. O BH é preferível ao controle de FWER (Bonferroni, Holm) em descoberta causal porque aceita alguns falsos positivos em troca de maior poder de detecção — apropriado para grafos esparsos onde cada aresta verdadeira tem valor informativo (Benjamini & Hochberg, 1995).

É fundamental distinguir dois tipos de valores produzidos pelo módulo:
- **p-values brutos** (`edge_pvalues`): retidos apenas para fins de transparência e debug, **nunca usados para calibração de priors**
- **q-values BH-corrigidos** (`edge_qvalues`): consumidos pelo módulo de calibração; interpretáveis como P(H₀ | dados) sob o framework Empirical Bayes (Seção 6.4)

### 6.3.5 Consenso entre Geos

Para cada geo disponível (máximo 5, amostrados aleatoriamente se mais disponíveis), executa-se o PCMCI independentemente, produzindo um grafo por geo. O **grafo de consenso** é construído por maioria de votos: uma aresta i → j é incluída se detectada em ≥ 50% dos geos. Os q-values de consenso são médias condicionais sobre os geos que detectaram a aresta, preservando a interpretação Empirical Bayes.

Este mecanismo oferece robustez contra ruído geo-específico: uma aresta causal real, presente em múltiplos geos, acumula evidência independente e sobrevive ao consenso; uma aresta espúria, detectada em apenas um geo por acaso, é suprimida.

### 6.3.6 Restrições Estruturais Específicas de MMM

Conhecimento de domínio sobre MMM é codificado como restrições que eliminam arestas estruturalmente impossíveis antes da busca:

- **y não causa nada:** por precedência temporal, a variável de resposta não pode causar canais ou controles
- **Controles são exógenos:** variáveis de controle (preço, sazonalidade) são determinadas por fatores externos; canais de marketing não causam preço

Essas restrições são passadas ao PCMCI via `link_assumptions`, reduzindo o espaço de busca e melhorando o poder de detecção das arestas remanescentes.
```

- [ ] **Passo 4: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 6.3 — descoberta estrutural via CD-NOTS com justificativa"
```

---

## Task 5: Escrever Seção 6.4 — Empirical Bayes e Calibração de Priors

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 6.4)

- [ ] **Passo 1: Escrever subseções 6.4.1 a 6.4.3 (derivação formal)**

```markdown
## 6.4 Tradução do Grafo em Priors Estruturais (Empirical Bayes)

### 6.4.1 Fundamento: q-value como Aproximação de P(H₀ | dados)

A ponte formal entre a descoberta causal (Seção 6.3) e a inferência Bayesiana (Seção 6.5) é estabelecida pelo framework de Empirical Bayes para testes múltiplos (Storey, 2002; Efron, 2010, Cap. 5).

Sob este framework, o q-value de Benjamini-Hochberg satisfaz:

$$q_i \approx P(H_0 \mid \text{dados}_i)$$

onde $H_0$ é a hipótese nula de ausência de aresta causal. Portanto, a **probabilidade de inclusão posterior (PIP)** — a probabilidade de que o efeito causal seja real dado os dados observados — é:

$$\text{PIP}_i = 1 - q_i \approx P(H_A \mid \text{dados}_i) = P(\text{efeito causal real} \mid \text{dados})$$

Esta identidade é a contribuição central deste estágio do pipeline: ela conecta formalmente um output de descoberta causal (q-value, calculado por PCMCI + BH) com uma quantidade diretamente interpretável no paradigma Bayesiano (PIP). Um canal com q = 0,01 tem PIP = 0,99 — há 99% de probabilidade posterior de que o efeito causal seja real, e o prior deve permitir que o modelo estime livremente sua magnitude. Um canal excluído com q = 0,99 tem PIP = 0,01 — a evidência é forte contra a presença de efeito, e o prior deve regularizar fortemente para zero.

### 6.4.2 Prior Ideal: Mistura Spike-and-Slab

O prior ideal para o coeficiente β de canal é a mistura discreta (Ishwaran & Rao, 2005):

$$\beta \sim \text{PIP} \times \text{HalfNormal}(\sigma_{\text{base}}) + (1 - \text{PIP}) \times \delta(0)$$

onde δ(0) é a massa pontual em zero. Esta especificação formaliza a ideia de que β tem probabilidade PIP de ser um valor positivo (com dispersão σ_base) e probabilidade (1 − PIP) de ser exatamente zero (canal sem efeito).

A especificação discreta é intratável em MCMC porque requer variáveis latentes binárias que causam convergência lenta e dificuldade de exploração do espaço posterior.

### 6.4.3 Relaxação Contínua: A Fórmula Central

Emprega-se a **relaxação contínua do spike-and-slab** (Ishwaran & Rao, 2005), que substitui δ(0) por uma HalfNormal de largura mínima e interpola linearmente entre os extremos:

$$\sigma_{\text{adj}} = \sigma_{\text{base}} \times \left(\underbrace{\text{MIN\_SIGMA\_RATIO}}_{\text{spike}} + \underbrace{(1 - \text{MIN\_SIGMA\_RATIO})}_{\text{escala}} \times \text{PIP}\right)$$

Com MIN_SIGMA_RATIO = 0,4:

$$\sigma_{\text{adj}} = \sigma_{\text{base}} \times (0{,}4 + 0{,}6 \times (1 - q))$$

**Propriedades da interpolação:**
- PIP = 1 (q ≈ 0): $\sigma_{\text{adj}} = \sigma_{\text{base}} \times 1{,}0$ — prior máximo, dados decidem livremente
- PIP = 0 (q ≈ 1): $\sigma_{\text{adj}} = \sigma_{\text{base}} \times 0{,}4$ — máxima regularização, prior empurra para zero
```

- [ ] **Passo 2: Escrever subseções 6.4.4 a 6.4.7**

```markdown
### 6.4.4 Derivação de MIN_SIGMA_RATIO = 0,4

A constante MIN_SIGMA_RATIO não é uma escolha arbitrária. Representa a largura mínima do componente spike da relaxação contínua, e sua escolha foi determinada empiricamente durante o desenvolvimento do pipeline: valores abaixo de 0,3 causam conflito prior-verossimilhança, manifestado como R-hat > 1,8 no diagnóstico MCMC — indicativo de que o prior excessivamente estreito impede o sampler de explorar adequadamente o espaço posterior. O valor 0,4 é o mínimo que preserva a tratabilidade do sampler NUTS em ambos os frameworks testados. O complemento (1 − 0,4) = 0,6 é consequência algébrica deste valor, não um parâmetro independente.

A tabela abaixo ilustra os multiplicadores para valores típicos de q-value:

| Tipo de canal | q-value | PIP = 1−q | σ_adj / σ_base |
|---------------|---------|-----------|----------------|
| Direto, evidência forte | 0,01 | 0,99 | ≈ 1,00 |
| Direto, evidência moderada | 0,20 | 0,80 | 0,88 |
| Mediado, borderline | 0,50 | 0,50 | 0,70 |
| Excluído, evidência fraca | 0,20 | 0,80 | 0,88 |
| Excluído, evidência confiante | 0,90 | 0,10 | 0,46 |
| Excluído, evidência muito confiante | 0,99 | 0,01 | ≈ 0,40 |

A **assimetria é conceitualmente correta**: quando a evidência de exclusão é fraca (q = 0,20), o multiplicador é 0,88 — quase sem regularização, pois a incerteza é alta e o modelo deve deferir aos dados. Apenas quando a exclusão é confiante (q ≥ 0,90) o prior exerce regularização substancial.

### 6.4.5 Canais Mediados: q-value do Elo Mais Fraco

Para canais que alcançam y somente por mediação (ch → ... → y, sem aresta direta), a confiança no efeito é limitada pelo elo mais fraco do caminho. Utiliza-se busca BFS min-max: entre todos os caminhos ch → ... → y com profundidade ≤ 3, seleciona-se aquele com menor q-value máximo (mínima incerteza no elo mais fraco). Este q-value é então usado na fórmula central da Seção 6.4.3.

### 6.4.6 Ajuste do Prior de Adstock: Beta(1, α_b)

O decaimento geométrico do adstock utiliza prior Beta(1, α_b), onde valores altos de α_b concentram a distribuição próxima de zero (decaimento rápido). O PIP modula α_b linearmente:

$$\alpha_b = \text{clip}(3{,}0 - 2{,}0 \times \text{PIP},\quad \text{min}=1{,}0,\quad \text{max}=5{,}0)$$

- Canal com PIP ≈ 1: α_b ≈ 1,0 → Beta(1,1) = Uniforme (adstock muito flexível)
- Canal com PIP ≈ 0: α_b ≈ 3,0 → Beta(1,3) (prior de decaimento rápido)

Canais causalmente confirmados recebem priors de adstock mais flexíveis porque há evidência de que seu efeito persiste no tempo; canais excluídos recebem prior de decaimento rápido porque, na ausência de efeito causal, qualquer persistência aparente é mais provavelmente ruído.

### 6.4.7 Penalidade de Endogeneidade

Para canais detectados como confundidos por variável de controle (controle → canal no grafo), aplica-se penalidade proporcional ao R² da confundência:

$$\text{tolerance} = \max(\text{MIN\_SIGMA\_RATIO},\; 1 - R^2)$$
$$\text{multiplier\_final} = \max(\text{MIN\_SIGMA\_RATIO},\; \text{multiplier\_PIP} \times \text{tolerance})$$

O R² é calculado como o quadrado da correlação de Pearson entre a série do controle e a série do canal nos dados observados, fornecendo medida data-driven do grau de confundimento. Quanto maior a proporção de variância do canal explicada pelo controle, menor a confiança na estimativa de efetividade do canal, e menor o multiplicador final — sempre respeitando o piso MIN_SIGMA_RATIO = 0,4.
```

- [ ] **Passo 3: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 6.4 — Empirical Bayes, spike-and-slab, derivação PIP=1-q"
```

---

## Task 6: Escrever Seção 6.5 — Frameworks Bayesianos e Framework de Comparação

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 6.5)

- [ ] **Passo 1: Escrever subseções 6.5.1 e 6.5.2**

```markdown
## 6.5 Frameworks Bayesianos e o Framework de Comparação

### 6.5.1 PyMC-Marketing

O PyMC-Marketing (Abril et al., 2023) é um framework Bayesiano completo baseado em PyMC, com modelagem hierárquica geo-level, adstock geométrico e saturação Hill. O prior de cada canal é especificado via `HalfNormal(sigma=σ_adj)` para o coeficiente de efetividade (`beta_channel`), onde σ_adj é o desvio-padrão ajustado pelo multiplicador Empirical Bayes (Seção 6.4). O prior de adstock é especificado via `Beta(alpha=1, beta=α_b)`, onde α_b é ajustado pelo PIP (Seção 6.4.6). A estimação utiliza o sampler NUTS com backend nutpie/JAX para máxima performance computacional.

### 6.5.2 Meridian (Google)

O Meridian (Jin et al., 2024) é o framework Bayesiano geo-level do Google, implementado em TensorFlow Probability. Utiliza adstock Weibull CDF, que permite modelar picos de efeito deslocados no tempo — mais flexível que o decaimento geométrico puro. O prior de efetividade de canal é especificado via `LogNormal(mu, sigma=σ_adj)`. A estimação usa MCMC via TFP com configuração `n_adapt + n_burnin = n_tune` e `n_keep = n_draws`.

### 6.5.3 Pontos de Injeção dos Priors Calibrados

A modularidade de ambos os frameworks permite injetar os priors calibrados sem modificar a lógica interna de estimação:

- **PyMC-Marketing:** o argumento `sigma` do prior `HalfNormal` do `beta_channel` é substituído por `σ_adj` por canal; o argumento `beta` do prior `Beta` do adstock é substituído por `α_b` por canal
- **Meridian:** o parâmetro de dispersão do prior `LogNormal` de `beta_m` é substituído por `σ_adj` por canal via o objeto `PriorDistribution`

### 6.5.4 Framework de Comparação: `mmm_param_recovery`

O framework de benchmark `mmm_param_recovery` (repositório sibling) fornece a infraestrutura para comparação padronizada entre os 4 braços. Ele inclui: gerador de dados sintéticos, fitting dos Braços 1 e 2 (baselines PyMC e Meridian), cálculo unificado de métricas, e logging estruturado. Os Braços 3 e 4 são integrados via patches documentados em `CDNOTS_INTEGRATION.py` (descrito na Seção 7.6), sem modificar os braços baseline.

A garantia de comparabilidade é absoluta: todos os 4 braços recebem exatamente os mesmos dados de treinamento e avaliação, os mesmos hiperparâmetros MCMC (número de chains, draws, tune steps, target_accept), e a mesma semente aleatória. A única variável entre os braços é a especificação dos priors — padrão (Braços 1 e 2) versus calibrados pelo grafo CD-NOTS (Braços 3 e 4).
```

- [ ] **Passo 2: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 6.5 — frameworks Bayesianos e framework de comparação"
```

---

## Task 7: Escrever Seção 6.6 — Métricas de Avaliação

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 6.6)

- [ ] **Passo 1: Escrever as 4 dimensões de avaliação**

```markdown
## 6.6 Métricas de Avaliação

A avaliação do pipeline é organizada em quatro dimensões, cada uma respondendo a uma questão específica do experimento.

### 6.6.1 Estrutura Causal — "O algoritmo descobriu o grafo correto?"

Métricas calculadas pela comparação entre o grafo descoberto pelo CD-NOTS e a matriz de adjacência do ground truth (`_build_causal_ground_truth()`):

- **SHD** (Structural Hamming Distance) = |FP| + |FN|: número total de arestas incorretas (extras ou faltantes) em relação ao ground truth. Menor é melhor.
- **Precision** = TP / (TP + FP): fração das arestas descobertas que são verdadeiras. Alta Precision indica poucas arestas espúrias.
- **Recall** = TP / (TP + FN): fração das arestas verdadeiras que foram descobertas. Alto Recall indica poucos falsos negativos.
- **F1** = 2 × Precision × Recall / (Precision + Recall): média harmônica de Precision e Recall.
- **FDR** = FP / (TP + FP): taxa de falsas descobertas; complemento de Precision.

Critérios de sucesso para o preset `causal_business`: FDR ≤ 0,40, Precision ≥ 0,60, `ci_test_used == "parcorr"`.

### 6.6.2 Predição — "O modelo ajustado prevê melhor?"

- **R²**: coeficiente de determinação, por geo e agregado
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Square Error
- **Durbin-Watson**: estatística de autocorrelação residual (diagnóstico de especificação)

### 6.6.3 Atribuição — "O modelo recupera o impacto real de cada canal?"

- **ROAS estimado vs. ROAS verdadeiro** por canal: correlação de Pearson e RMSE
- **Correlação de contribuições**: Pearson entre contribuições estimadas e verdadeiras por canal
- **Ghost channels**: ROAS estimado deve ser ≈ 0 para Ghost-A e Ghost-B em todos os braços (teste de supressão)

### 6.6.4 Diagnósticos MCMC — "O sampler convergiu sem conflito prior-verossimilhança?"

- **R-hat < 1,05** para todos os parâmetros do modelo. O limiar 1,05 é mais restritivo que o convencional 1,1 e serve como diagnóstico específico de conflito prior-verossimilhança — o problema central que motivou o redesign do módulo de calibração
- **ESS mínimo** (Effective Sample Size): garante que as cadeias MCMC produziram amostras efetivamente independentes
```

- [ ] **Passo 2: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 6.6 — métricas de avaliação em 4 dimensões"
```

---

## Task 8: Escrever Seção 7.1 — Arquitetura do Sistema

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 7.1)

- [ ] **Passo 1: Escrever diagrama de dependências e descrição dos módulos**

```markdown
# 7. Implementação

## 7.1 Arquitetura do Sistema

O pipeline é implementado em cinco módulos Python com dependências estritamente sequenciais:

```
config.py + presets.py
(dataclasses e configurações concretas, incluindo causal_business)
     │
     ▼
[Dataset Sintético]
(gerado via mmm_param_recovery/benchmarking/data_generator.py)
     │
     ▼
cdnots_discovery.py  →  CausalGraph
(PCMCI + BH + consenso + restrições MMM)
     │
     ▼
cdnots_model_builder.py  →  multipliers[], adstock_params[]
(Empirical Bayes spike-and-slab por canal)
     │
     ├────────────────────────────────────────────────┐
     ▼                                                ▼
build_pymc_model_with_graph()          build_meridian_model_with_graph()
(Braço 3)                              (Braço 4)
     │                                                │
     └────────────────────┬───────────────────────────┘
                          ▼
                  cdnots_fitter.py
         (fit_pymc_with_graph, fit_meridian_with_graph)
                          │
                          ▼
              experimento_4bracos.ipynb
          (orquestração + avaliação + resultados)
```

Os módulos externos do repositório sibling `mmm_param_recovery` são consumidos como dependências:
- `benchmarking/model_builder.py`: calcula `prior_sigma` baseline a partir da escala dos dados
- `benchmarking/model_fitter.py`: fitting dos Braços 1 e 2 (sem modificação)
- `benchmarking/diagnostics.py`: `compute_ess()`, R-hat
- `benchmarking/evaluator.py`: ROAS por canal, contribuições, métricas de atribuição
```

- [ ] **Passo 2: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 7.1 — arquitetura do sistema"
```

---

## Task 9: Escrever Seção 7.2 — Geração de Dados

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 7.2)
- Reference: `arquivos_recentes/config.py`, `arquivos_recentes/presets.py`

- [ ] **Passo 1: Escrever seção 7.2 com trechos de código**

```markdown
## 7.2 Geração de Dados: `config.py` e `presets.py`

### 7.2.1 Especificação de Arestas Causais

O dataclass `CausalEdgeConfig` (`config.py`) especifica cada relação inter-canal do ground truth:

```python
@dataclass
class CausalEdgeConfig:
    source_channel: str   # canal de origem do spillover
    target_channel: str   # canal que recebe o spillover
    effect_size: float    # fração do spend adstockado do source propagada ao target
    lag: int              # defasagem em períodos antes do efeito
    decay: float          # taxa de decaimento geométrico do source antes do spillover
```

### 7.2.2 Preset `causal_business`

As arestas causais do preset são definidas em `presets.py`:

```python
causal_edges=[
    CausalEdgeConfig(
        source_channel="TV",
        target_channel="Search-Ads",
        effect_size=0.20,
        lag=2,
        decay=0.5,
    ),
    CausalEdgeConfig(
        source_channel="Social-Media",
        target_channel="Brand-Search",
        effect_size=0.15,
        lag=1,
        decay=0.4,
    ),
    CausalEdgeConfig(
        source_channel="Video",
        target_channel="Social-Media",
        effect_size=0.10,
        lag=1,
        decay=0.3,
    ),
],
```

Os canais ghost têm `base_effectiveness=0.0` mas padrões de gasto estatisticamente plausíveis:

```python
ChannelConfig(
    name="Ghost-A",
    pattern="seasonal",
    base_spend=2000.0,
    seasonal_amplitude=0.3,
    spend_volatility=0.25,
    base_effectiveness=0.0,  # zero efeito real nas vendas
),
```

### 7.2.3 Geração do Spillover Causal

A geração de dados opera em duas fases. Na Fase 1, cada canal gera seu spend base independentemente. Na Fase 2, o spillover causal é aplicado sequencialmente:

```python
# Fase 2: Spillover causal entre canais (loop sobre CausalEdgeConfig)
for edge in causal_edges:
    adstocked = np.zeros(n_periods)
    source = channel_spends[edge.source_channel]
    for t in range(n_periods):
        adstocked[t] = source[t] + edge.decay * (adstocked[t-1] if t > 0 else 0.0)
        if t >= edge.lag:
            channel_spends[edge.target_channel][t] += (
                edge.effect_size * adstocked[t - edge.lag]
            )
```

Efeitos mediados (Video → Social-Media → Brand-Search) emergem composicionalmente: o spillover de Video aumenta o spend de Social-Media, que então gera spillover adicional para Brand-Search na iteração seguinte do loop.
```

- [ ] **Passo 2: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 7.2 — geração de dados com código anotado"
```

---

## Task 10: Escrever Seção 7.3 — Módulo de Descoberta: `cdnots_discovery.py`

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 7.3)
- Reference: `arquivos_recentes/cdnots_discovery.py`

- [ ] **Passo 1: Escrever subseções 7.3.1 e 7.3.2**

```markdown
## 7.3 Módulo de Descoberta Causal: `cdnots_discovery.py`

### 7.3.1 Estrutura de Dados: `CausalGraph`

O resultado da descoberta é encapsulado no dataclass imutável `CausalGraph`:

```python
@dataclass(frozen=True)
class CausalGraph:
    adjacency_matrix: np.ndarray      # adj[i,j]=1 se aresta i→j no consenso
    edge_pvalues: np.ndarray          # p-values MCI brutos (para debug)
    edge_qvalues: np.ndarray          # q-values BH-corrigidos (para calibração)
    variable_names: Tuple[str, ...]   # canais + controles + "y"
    direct_channels: Tuple[str, ...]  # aresta direta ch→y detectada
    mediated_channels: Tuple[str, ...] # caminho ch→...→y sem aresta direta
    excluded_channels: Tuple[str, ...]  # sem caminho para y
    endogenous_channels: Tuple[str, ...] # confundidos por controles
    endogenous_r2: Dict[str, float]     # R² da confundência por canal
    ci_test_used: str                 # "parcorr" ou "kci" (para auditabilidade)
```

A distinção entre `edge_pvalues` e `edge_qvalues` é invariante de design: apenas os q-values BH-corrigidos são consumidos pelo módulo de calibração; os p-values brutos são retidos exclusivamente para debug e transparência.

### 7.3.2 Função Principal: `discover_graph`

A assinatura e a seleção adaptativa do CI test:

```python
def discover_graph(
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: Optional[List[str]] = None,
    alpha: float = 0.05,
    max_lag: int = 1,
    max_geos: int = 5,
    ci_test: str = "auto",  # "auto" | "parcorr" | "kci"
    console: Optional[Console] = None,
    max_conds_dim: Optional[int] = None,
) -> CausalGraph:
    ...
    # multi-geo → parcorr (linear, 100-1000× mais rápido que kernel)
    # single-geo → kci (não-paramétrico, captura adstock/saturação)
    if ci_test == "auto":
        ci_test = "kci" if len(geos) == 1 else "parcorr"
```
```

- [ ] **Passo 2: Escrever subseções 7.3.3 a 7.3.5**

```markdown
### 7.3.3 Execução do PCMCI com Correção BH

Para cada geo, o PCMCI é executado com as configurações:

```python
results = pcmci.run_pcmci(
    tau_max=max_lag,
    tau_min=1,                    # apenas relações defasadas; contemporâneas são ambíguas
    pc_alpha=alpha,
    max_conds_dim=max_conds_dim,  # limita explosão combinatória dos conjuntos de condicionamento
    link_assumptions=link_assumptions,  # restrições estruturais MMM
)
p_matrix = results["p_matrix"]

# Correção BH sobre todos os pares (i,j,τ) simultaneamente
q_matrix = pcmci.get_corrected_pvalues(
    p_matrix=p_matrix, tau_min=1, tau_max=max_lag, fdr_method="fdr_bh"
)

# Inclusão: aresta i→j presente se q < α em qualquer defasagem τ
for i in range(n_vars):
    for j in range(n_vars):
        adj[i, j] = 1 if np.min(q_matrix[i, j, 1:max_lag+1]) < alpha else 0
```

### 7.3.4 Consenso entre Geos e Restrições Estruturais

```python
# Consenso por maioria de votos
stacked = np.stack(list(per_geo_adj.values()), axis=0)  # (n_geos, n_vars, n_vars)
agreement = stacked.mean(axis=0)
consensus_adj = (agreement >= 0.5).astype(float)

# Restrições estruturais: y não causa nada; canais não causam controles
consensus_adj[y_idx, :] = 0
for ci in channel_indices:
    for cj in control_indices:
        consensus_adj[ci, cj] = 0
```

### 7.3.5 Restrições Estruturais via `link_assumptions`

```python
def _build_mmm_link_assumptions(n_channels, n_controls, n_vars, max_lag):
    """Restringe o PCMCI ao espaço estruturalmente válido para MMM.

    Permitidas: canal → y, controle → y, canal → canal, controle → canal
    Proibidas:  y → qualquer coisa, canal → controle, controle → controle
    """
    # Retorna dict no formato esperado por tigramite:
    # {target_node: {(source_node, -lag): link_type}}
    # "" = proibido, "-?>" = possível com defasagem, "o?>" = contemporâneo permitido
    ...
```
```

- [ ] **Passo 3: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 7.3 — módulo cdnots_discovery.py com trechos anotados"
```

---

## Task 11: Escrever Seção 7.4 — Módulo de Calibração: `cdnots_model_builder.py`

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 7.4)
- Reference: `arquivos_recentes/cdnots_model_builder.py`

- [ ] **Passo 1: Escrever seção 7.4 com os três trechos de código principais**

```markdown
## 7.4 Módulo de Calibração de Priors: `cdnots_model_builder.py`

### 7.4.1 Cálculo dos Multiplicadores: `_compute_sigma_multipliers`

```python
MIN_SIGMA_RATIO = 0.4  # spike mínimo; valores < 0.3 causam R-hat > 1.8

def _compute_sigma_multipliers(channel_columns, graph):
    multipliers = np.ones(len(channel_columns))
    y_idx = graph.variable_names.index("y")

    for i, ch in enumerate(channel_columns):
        ch_idx = graph.variable_names.index(ch)
        has_direct = graph.adjacency_matrix[ch_idx, y_idx] > 0

        if has_direct:
            q_value = float(graph.edge_qvalues[ch_idx, y_idx])
        elif ch in graph.mediated_channels:
            # q-value do elo mais fraco no caminho mais confiante até y
            q_value = _path_min_confidence_pvalue(
                graph.adjacency_matrix, graph.edge_qvalues, ch_idx, y_idx
            )
        else:  # canal excluído
            q_value = float(graph.edge_qvalues[ch_idx, y_idx])  # alto por definição

        pip = float(np.clip(1.0 - q_value, 0.0, 1.0))
        multipliers[i] = MIN_SIGMA_RATIO + (1.0 - MIN_SIGMA_RATIO) * pip

        # Penalidade de endogeneidade: shrinkage adicional proporcional a R²
        if ch in graph.endogenous_r2:
            r2 = graph.endogenous_r2[ch]
            tolerance = max(MIN_SIGMA_RATIO, 1.0 - r2)
            multipliers[i] = max(MIN_SIGMA_RATIO, multipliers[i] * tolerance)

    return multipliers
```

### 7.4.2 Parâmetros de Adstock: `_compute_adstock_params`

```python
def _compute_adstock_params(channel_columns, graph):
    BASE_B, MIN_B = 3.0, 1.0
    alpha_b = np.full(len(channel_columns), BASE_B)

    for i, ch in enumerate(channel_columns):
        # ... obter q_value pelo mesmo padrão de _compute_sigma_multipliers
        pip = float(np.clip(1.0 - q_value, 0.0, 1.0))
        # PIP alto → alpha_b baixo → Beta mais uniforme → adstock mais flexível
        alpha_b[i] = float(np.clip(BASE_B - (BASE_B - MIN_B) * pip, MIN_B, 5.0))

    return np.ones(len(channel_columns)), alpha_b  # (alpha_a=1 fixo, alpha_b variável)
```

### 7.4.3 BFS Min-Max para Canais Mediados: `_path_min_confidence_pvalue`

```python
def _path_min_confidence_pvalue(adj, pvals, source, target, max_depth=3):
    """Retorna o q-value do elo mais fraco no caminho mais confiante de source a target."""
    best = {source: 0.0}  # menor q-value máximo observado até o nó
    frontier = [source]
    for _ in range(max_depth):
        next_frontier = []
        for node in frontier:
            for child in range(adj.shape[1]):
                if adj[node, child] <= 0 or child == node:
                    continue
                # Bottleneck: caminho é tão bom quanto seu elo mais fraco
                path_max = max(best[node], float(pvals[node, child]))
                if path_max < best.get(child, np.inf):
                    best[child] = path_max
                    next_frontier.append(child)
        frontier = next_frontier
    return best.get(target, 1.0)  # 1.0 = nenhum caminho encontrado → máxima incerteza
```
```

- [ ] **Passo 2: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 7.4 — módulo cdnots_model_builder.py com trechos anotados"
```

---

## Task 12: Escrever Seção 7.5 — Ajuste dos Modelos: `cdnots_fitter.py`

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seção 7.5)
- Reference: `arquivos_recentes/cdnots_fitter.py`

- [ ] **Passo 1: Escrever seção 7.5 com trechos de build e fit**

```markdown
## 7.5 Ajuste dos Modelos: `cdnots_fitter.py`

### 7.5.1 Construção do Modelo PyMC com Priors Calibrados

```python
def build_pymc_model_with_graph(data_df, channel_columns, control_columns, graph):
    # Prior base proporcional à escala dos dados (herdado do baseline)
    prior_sigma = model_builder.calculate_prior_sigma(data_df, channel_columns)

    # Multiplicadores Empirical Bayes por canal: array shape (n_channels,)
    multipliers = _compute_sigma_multipliers(channel_columns, graph)
    adjusted_sigma = prior_sigma * multipliers[np.newaxis, :]  # broadcast para (geo, channel)

    # Saturação com sigma calibrado por canal
    saturation = HillSaturationSigmoid(
        priors={
            "sigma": Prior("HalfNormal", sigma=adjusted_sigma.mean(axis=0), dims=("channel",)),
        },
    )

    # Adstock geométrico com Beta(1, alpha_b) calibrado por canal
    _, alpha_b = _compute_adstock_params(channel_columns, graph)
    adstock = GeometricAdstock(
        l_max=8,
        priors={"alpha": Prior("Beta", alpha=1, beta=alpha_b.tolist(), dims=("channel",))},
    )

    return MMM(
        date_column="time", target_column="y",
        channel_columns=channel_columns, control_columns=control_columns,
        saturation=saturation, adstock=adstock,
        yearly_seasonality=2,
    )
```

### 7.5.2 Fitting do Braço 3 (PyMC + CD-NOTS)

```python
def fit_pymc_with_graph(data_df, channel_columns, control_columns, graph,
                         sampler, n_chains, n_draws, n_tune, target_accept, seed):
    start = time.perf_counter()

    pymc_model = build_pymc_model_with_graph(
        data_df, channel_columns, control_columns, graph
    )
    pymc_model.fit(
        X=data_df.drop(columns=["y"]), y=data_df["y"],
        chains=n_chains, draws=n_draws, tune=n_tune,
        target_accept=target_accept, random_seed=seed,
        nuts_sampler=sampler,  # "nutpie" usa backend JAX para máxima performance
    )
    pymc_model.sample_posterior_predictive(
        X=data_df.drop(columns=["y"]), extend_idata=True, random_seed=seed
    )

    runtime = time.perf_counter() - start
    ess = diagnostics.compute_ess(pymc_model.idata)
    return pymc_model, runtime, ess  # mesma interface que model_fitter.fit_pymc
```

O Braço 4 (Meridian + CD-NOTS) segue o mesmo padrão via `build_meridian_model_with_graph` e `fit_meridian_with_graph`, com ajuste do prior `LogNormal` e uso do sampler TFP.
```

- [ ] **Passo 2: Commit**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seção 7.5 — cdnots_fitter.py com build e fit anotados"
```

---

## Task 13: Escrever Seções 7.6 e 7.7 — Benchmark e Reprodutibilidade

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md` (seções 7.6-7.7)
- Reference: `arquivos_recentes/CDNOTS_INTEGRATION.py`

- [ ] **Passo 1: Escrever seção 7.6 — Framework de benchmark e notebook**

```markdown
## 7.6 Framework de Benchmark e Notebook

### 7.6.1 Estrutura do Repositório `mmm_param_recovery`

```
mmm_param_recovery/
├── benchmarking/
│   ├── config.py          # MMMDataConfig, ChannelConfig, CausalEdgeConfig
│   ├── presets.py         # causal_business e outros presets
│   ├── data_generator.py  # geração de dados sintéticos com spillover causal
│   ├── model_builder.py   # build_pymc_model, build_meridian_model (Braços 1 e 2)
│   ├── model_fitter.py    # fit_pymc, fit_meridian (Braços 1 e 2)
│   ├── diagnostics.py     # compute_ess(), r_hat()
│   └── evaluator.py       # ROAS por canal, contribuições, métricas de atribuição
└── run_benchmark.py       # orquestração da execução dos 4 braços
```

### 7.6.2 Integração dos Braços 3 e 4 via `CDNOTS_INTEGRATION.py`

O arquivo `CDNOTS_INTEGRATION.py` documenta os patches a serem aplicados em `run_benchmark.py` para adicionar os Braços 3 e 4 sem alterar o código dos braços baseline:

```python
# Descoberta do grafo (executada uma vez, compartilhada pelos Braços 3 e 4)
graph = cdnots_discovery.discover_graph(
    data_df, channel_columns, control_columns=control_columns,
    alpha=0.05, max_lag=2, console=console
)

# Braço 3: PyMC + CD-NOTS
pymc_cdnots, runtime_3, ess_3 = cdnots_fitter.fit_pymc_with_graph(
    data_df, channel_columns, control_columns, graph,
    sampler=args.sampler, n_chains=args.chains, n_draws=args.draws,
    n_tune=args.tune, target_accept=args.target_accept, seed=args.seed
)

# Braço 4: Meridian + CD-NOTS
meridian_cdnots, runtime_4, ess_4 = cdnots_fitter.fit_meridian_with_graph(
    data_df, channel_columns, control_columns, graph,
    n_chains=args.chains, n_draws=args.draws, n_tune=args.tune,
    target_accept=args.target_accept, seed=args.seed
)
```

### 7.6.3 Notebook `experimento_4bracos.ipynb`

O notebook organiza a execução dos 4 braços em células sequenciais:
1. **Setup:** imports, carregamento do preset `causal_business`, definição do seed global
2. **Descoberta:** `discover_graph()` + visualização do grafo de consenso + métricas estruturais (SHD, Precision, FDR)
3. **Braços 1–4:** fitting sequencial com timing e logging de R-hat / ESS
4. **Avaliação:** tabelas comparativas de R², MAPE, ROAS por canal para os 4 braços
5. **Diagnóstico:** heatmaps de R-hat por parâmetro e braço
```

- [ ] **Passo 2: Escrever seção 7.7 — Reprodutibilidade**

```markdown
## 7.7 Reprodutibilidade

O experimento é completamente reproduzível:

**Ambiente:**
```bash
pip install -e ".[cdnots,viz,dev]"  # instala causalmmm + extras necessários
```

Dependências principais: Python 3.10+, tigramite ≥ 0.7, pymc-marketing ≥ 0.9, meridian (Google), tensorflow-probability.

**Seeds:** o preset usa `seed=2025_07_15` por padrão, propagada para o gerador de dados e para os samplers MCMC de todos os braços.

**Execução:** o notebook `experimento_4bracos.ipynb` pode ser executado via Jupyter. Para execução em ambiente remoto (GCP), o script `pack_for_gcp.sh` empacota ambos os repositórios para upload ao GCS. O guia `COMO_EXECUTAR.md` documenta o fluxo completo com `tmux` e checkpointing de resultados em `resultados/`.
```

- [ ] **Passo 3: Commit final do Capítulo 7**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: seções 7.6-7.7 — benchmark, notebook e reprodutibilidade"
```

---

## Task 14: Verificação Final e Limpeza

**Files:**
- Modify: `arquivos_recentes/dissertacao_relatorio_tecnico_completo.md`

- [ ] **Passo 1: Buscar placeholders remanescentes**

```bash
grep -n "\[INSERIR\]\|\[RASCUNHO\]\|TODO\|PLACEHOLDER" \
    arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
```

Esperado: zero ocorrências nas seções 6 e 7.

- [ ] **Passo 2: Verificar consistência numérica**

```bash
grep -n "MIN_SIGMA_RATIO\|0\.4\|BASE_B\|3\.0\|MIN_B\|1\.0" \
    arquivos_recentes/dissertacao_relatorio_tecnico_completo.md | head -30
```

Confirmar que MIN_SIGMA_RATIO = 0,4 (e não 0.3 ou 0.5), BASE_B = 3,0, MIN_B = 1,0.

- [ ] **Passo 3: Confirmar seções espelhadas**

Verificar manualmente que cada subseção 6.x tem correspondente 7.x:
- 6.2 ↔ 7.2 (dados sintéticos)
- 6.3 ↔ 7.3 (descoberta)
- 6.4 ↔ 7.4 (calibração)
- 6.5 ↔ 7.5 (frameworks)
- 6.6 ↔ 7.6 (benchmark/métricas)

- [ ] **Passo 4: Verificar que seções 9-14 continuam intactas**

```bash
grep -n "^# [0-9]\|^## [0-9]" \
    arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
```

Confirmar que o sumário mostra: 6, 7, 9 (Resultados), 10 (Impactos), 11, 12, 13, 14 (Referências).

- [ ] **Passo 5: Commit final**

```bash
git add arquivos_recentes/dissertacao_relatorio_tecnico_completo.md
git commit -m "doc: verificação final — capítulos 6 e 7 completos e consistentes"
```

---

## Self-Review

**Spec coverage:**
- 6.1 → Task 2 ✓
- 6.2 → Task 3 ✓
- 6.3 (incluindo justificativa CD-NOTS) → Task 4 ✓
- 6.4 (derivação PIP=1-q, MIN_SIGMA_RATIO) → Task 5 ✓
- 6.5 (PyMC, Meridian, mmm_param_recovery) → Task 6 ✓
- 6.6 (4 dimensões de avaliação) → Task 7 ✓
- 7.1 (arquitetura) → Task 8 ✓
- 7.2 (config.py, presets.py) → Task 9 ✓
- 7.3 (cdnots_discovery.py) → Task 10 ✓
- 7.4 (cdnots_model_builder.py) → Task 11 ✓
- 7.5 (cdnots_fitter.py) → Task 12 ✓
- 7.6-7.7 (benchmark, notebook, reprodutibilidade) → Task 13 ✓
- Verificação final → Task 14 ✓

**Placeholders:** nenhum TBD ou TODO nos trechos de código — todos extraídos dos arquivos reais.

**Consistência de tipos:** os trechos de código usam os mesmos nomes de funções e constantes em Tasks 5 e 11 (`MIN_SIGMA_RATIO`, `_compute_sigma_multipliers`, `_path_min_confidence_pvalue`), e as mesmas entre Tasks 6 e 12 (`build_pymc_model_with_graph`, `fit_pymc_with_graph`).
