# Universidade Presbiteriana Mackenzie

## Programa de Pós-Graduação em Computação Aplicada

**Rafael Silva Ennes**

# Investigação da Utilização de Descoberta de Estruturas Causais para Calibração de Modelos Bayesianos em Marketing Mix Modeling

São Paulo
2026

---

# Resumo

O Marketing Mix Modeling (MMM) é uma técnica amplamente utilizada para avaliar a eficácia de investimentos em diferentes canais de mídia e orientar decisões de alocação orçamentária. Os frameworks Bayesianos dominantes — PyMC-Marketing e Google Meridian — dependem de estruturas causais especificadas manualmente e priors calibrados por experimentos de incrementalidade, recursos inacessíveis à maioria das organizações. Paralelamente, abordagens de descoberta causal baseadas em redes neurais (CausalMMM) demonstram capacidade de identificar estruturas causais automaticamente, mas requerem volumes de dados (N > 100 entidades geográficas) incompatíveis com a realidade de mercados que operam com estratégias nacionais, como o brasileiro. Esta pesquisa propõe e avalia uma abordagem intermediária: a utilização do algoritmo CD-NOTS (Constraint-based Causal Discovery from Nonstationary Time-Series) para descoberta automática de grafos causais entre canais de marketing, e a tradução desses grafos em priors estruturais para calibração de modelos Bayesianos de MMM. O pipeline proposto — descoberta causal → tradução para priors → modelagem Bayesiana — é avaliado em design experimental de 4 braços comparando PyMC-Marketing e Meridian com e sem calibração via CD-NOTS, utilizando dados sintéticos com ground truth conhecido. Até onde sabemos, esta é a primeira proposta de integrar descoberta causal algorítmica baseada em testes de independência condicional como mecanismo de calibração estrutural de priors em modelos Bayesianos de Marketing Mix Modeling.

**Palavras-chave:** Marketing Mix Modeling; Descoberta Causal; CD-NOTS; Inferência Bayesiana; Priors Estruturais; PyMC-Marketing; Meridian.

---

# Abstract

[Versão em inglês do resumo — a ser redigida]

---

# Sumário

1. Contexto
2. Problema
3. Público-alvo
4. Fundamentação Teórica
   - 4.1 Marketing Mix Modeling: Origens, Conceitos e Evolução
   - 4.2 Efeitos Temporais e Não-Lineares em Marketing
   - 4.3 Frameworks Bayesianos para MMM
   - 4.4 Descoberta Causal em Séries Temporais
   - 4.5 Complementaridade entre Métodos Observacionais e Experimentais
5. Soluções Existentes (Technology Road Map)
   - 5.1 Estado da Prática: Frameworks Open-Source
   - 5.2 Estado da Arte: Descoberta Causal Aplicada a MMM
   - 5.3 Lacuna Identificada
6. Solução Proposta
   - 6.1 Pipeline: Descoberta Causal → Priors Estruturais → Modelagem Bayesiana
   - 6.2 Design Experimental de 4 Braços
   - 6.3 Métricas de Avaliação
7. Desenvolvimento (MVP)
   - 7.1 Módulo de Descoberta Causal (CD-NOTS)
   - 7.2 Módulo de Calibração de Priors (Empirical Bayes)
   - 7.3 Geração do Dataset com Estrutura Causal Conhecida
   - 7.4 Framework de Benchmark
8. Validação
9. Resultados
10. Impactos
11. Plano de Exploração da Tecnologia
12. Artefatos Tecnológicos Gerados
13. Publicações
14. Referências

---

# 1. Contexto

## 1.1 Investimentos em Marketing e a Necessidade de Mensuração Causal

As organizações contemporâneas alocam recursos financeiros substanciais em atividades de marketing, demandando instrumentos metodológicos rigorosos para quantificar o retorno desses investimentos. Segundo Hanssens (2024), o Return on Marketing Investment (ROMI) apresenta variação dramática conforme os níveis de investimento, comportamento atribuível aos padrões não-lineares que caracterizam a resposta das métricas de marketing às mudanças nos gastos alocados. Esta constatação revela que a relação entre investimento de marketing e resultado financeiro não segue função linear simplificada, exigindo modelagem estatística sofisticada para capturar adequadamente a dinâmica de resposta mercadológica.

A urgência em demonstrar o valor econômico do marketing consolidou-se como imperativo estratégico nas organizações modernas. Morgan et al. (2022) documentam que 83% dos líderes de marketing identificam a demonstração de ROI como prioridade máxima em suas agendas corporativas. Este dado evidencia não apenas uma preocupação corporativa isolada, mas sim uma tendência sistêmica onde a comprovação da efetividade do marketing torna-se condição essencial para justificar alocações orçamentárias e garantir continuidade de investimentos mercadológicos.

## 1.2 Limitações das Métricas Isoladas e Emergência da Mensuração Triangular

A utilização de métricas isoladas revela-se insuficiente para capturar a complexidade multidimensional do impacto mercadológico. Hanssens & Pauwels (2016) demonstram que métricas atitudinais, comportamentais e financeiras apresentam correlações fracas entre si, sugerindo que nenhuma dimensão isolada fornece representação adequada do fenômeno de interesse.

A fragilidade das abordagens unimodais de mensuração conduziu à emergência de frameworks integrados. Van der Weide et al. (2020) demonstraram empiricamente que a integração de Marketing Mix Modeling (MMM), Marketing Technology Attribution (MTA) e testes de incrementalidade em framework unificado produz aumento de 40% no uplift esperado comparativamente à utilização isolada de MTA.

Consequentemente, a evolução do pensamento em mensuração de marketing migra de paradigma fundado em métricas compartimentalizadas para paradigma fundamentado em triangulação metodológica sistemática. A convergência de evidências provenientes de múltiplas abordagens analíticas fornece fundação mais sólida para inferência causal e alocação de recursos.

## 1.3 MMM como Solução Comprovada

A transição conceitual de despesa para investimento em marketing fundamenta-se na evidência empírica de que ações mercadológicas geram efeitos persistentes que transcendem períodos imediatos de execução. Dekimpe & Hanssens (1995, 1999) estabeleceram que marketing cria efeitos persistentes em vendas. MMM captura efeitos de longo prazo, retornos decrescentes e sinergias entre canais que MTA não consegue mensurar.

A sinergia entre canais — por exemplo, aumento de efetividade da publicidade digital quando acompanhada de investimentos em branding tradicional — representa fonte significativa de impacto que permanece invisível em abordagens baseadas em atribuição de último touchpoint, introduzindo viés sistemático nas estimativas de efetividade individual de canais.

## 1.4 Democratização através de Frameworks Open-Source

Jin et al. (2017) estabeleceram fundações teóricas com transformações adstock e Hill. Meta lançou Robyn (2020), Google lançou LightweightMMM (2022) e Meridian (2024), democratizando técnicas anteriormente proprietárias. A disponibilização de frameworks open-source representa transição fundamental na estrutura de conhecimento de MMM, migrando de paradigma concentrado em propriedade intelectual corporativa para paradigma fundamentado em colaboração técnica aberta.

## 1.5 Superioridade de Métodos Bayesianos para Contextos de Marketing

Rossi & Allenby (2003) estabeleceram vantagens Bayesianas: modularidade, incorporação de conhecimento prévio, quantificação de incerteza. Wang et al. (2017) demonstraram como pooling hierárquico de dados de categoria melhora estimativas com amostras pequenas. Chen et al. (2021) demonstram que codificar conhecimento de domínio como restrições de sinal em priors Bayesianos melhora a qualidade da estimação.

A superioridade dos métodos Bayesianos em contexto de MMM representa alinhamento fundamental entre estrutura do paradigma estatístico e característica da realidade mercadológica que busca modelar.

## 1.6 Fronteira: Integração de Causalidade através de DAGs e Estruturas Aprendidas

CausalMMM (GONG et al., 2024) demonstra melhorias de 5,7-7,1% em AUROC descobrindo estruturas causais automaticamente usando Graph VAE. DeepCausalMMM (TIRUMALA, 2025) combina GRUs com DAG learning para padrões temporais complexos. Filippou et al. (2025) aplicam PCMCI a dados agregados de marketing, demonstrando atribuição causal viável sem dados de nível de usuário. Estas contribuições inauguram a aplicação direta de descoberta causal a dados de marketing, mas deixam em aberto a questão de como integrar grafos descobertos com os frameworks Bayesianos já estabelecidos na prática.

---

# 2. Problema

## 2.1 O Gap entre Descoberta Causal e Modelagem Bayesiana em MMM

Os frameworks Bayesianos dominantes (PyMC-Marketing, Meridian) assumem uma estrutura causal fixa e pré-definida pelo analista: canais → vendas, sem interações entre canais. O que os torna "causais" na narrativa do mercado é que os priors são calibrados a partir de experimentos de incrementalidade. Porém, como documentado por Zhang et al. (2024) do Google, Lewis & Rao (2015) no QJE, e Gordon et al. (2019) no Marketing Science, a calibração experimental enfrenta limitações severas: custo combinatório O(d²) para mapear todas as relações entre d canais, natureza estática dos resultados, incapacidade de capturar mediação inter-canal, e inacessibilidade para a maioria das organizações.

Paralelamente, métodos de descoberta causal baseados em redes neurais (CausalMMM, DeepCausalMMM) demonstram capacidade de descobrir estruturas causais automaticamente, mas são frameworks end-to-end de deep learning que **substituem** modelos Bayesianos em vez de **melhorá-los**. Além disso, conforme documentado por Gong et al. (2024, Figura 4), esses modelos apresentam degradação significativa de performance com N < 100 entidades — limitação crítica para sua adoção prática.

## 2.2 O Problema de Low-N e a Inviabilidade de Redes Neurais no Contexto Brasileiro

O CausalMMM foi validado com dados de plataformas de e-commerce chinesas (Alibaba), onde a heterogeneidade entre centenas de "shops" fornece variação suficiente para o Graph VAE aprender estruturas causais diferenciadas. Em mercados com distribuição geográfica ampla — como os Estados Unidos, com 50+ estados ou 200+ DMAs (Designated Market Areas) — a modelagem geo-level multiplica os datapoints disponíveis, viabilizando abordagens de deep learning.

No entanto, esta não é a realidade da maioria dos mercados. No Brasil, a grande maioria das empresas opera com **estratégias de mídia nacionais**: o mesmo investimento em TV aberta, as mesmas campanhas digitais, distribuídos uniformemente pelo território. Não há variação geo-level significativa para explorar. Uma empresa brasileira típica dispõe de 104-208 semanas de dados semanais com 1 geo (nacional) ou no máximo 5 regiões (Sul, Sudeste, Centro-Oeste, Nordeste, Norte) — totalizando 104-1040 observações. Este volume é insuficiente para treinar um Graph VAE com as arquiteturas propostas por Gong et al. (2024) ou Tirumala (2025).

Este cenário representa um problema mais amplo do que o caso brasileiro: qualquer mercado onde a estratégia de mídia é predominantemente nacional (muitos países europeus menores, mercados asiáticos não fragmentados geograficamente) enfrenta a mesma limitação. O avanço científico em descoberta causal via redes neurais corre o risco de permanecer restrito a contextos com abundância de dados geo-level — uma fração minoritária dos cenários reais de aplicação de MMM.

## 2.3 Formulação do Problema

Diante deste cenário, o problema de pesquisa pode ser formulado como:

**É possível melhorar a especificação de modelos Bayesianos de MMM (PyMC-Marketing, Meridian) utilizando grafos causais descobertos algoritmicamente via testes de independência condicional (CD-NOTS) como priors estruturais, em contextos de dados com poucos datapoints (low-N) onde abordagens de deep learning são inviáveis?**

Este problema desdobra-se em três questões específicas:

1. O CD-NOTS consegue descobrir estruturas causais informativas entre canais de marketing a partir de séries temporais com 104-208 observações?
2. A tradução do grafo causal descoberto em ajustes de priors Bayesianos melhora as métricas preditivas e de atribuição dos modelos PyMC-Marketing e Meridian?
3. Qual o ganho marginal da descoberta causal sobre os modelos baseline, e em quais cenários esse ganho é mais pronunciado?

---

# 3. Público-alvo

O artefato tecnológico desenvolvido nesta pesquisa destina-se a três públicos:

**Praticantes de MMM em empresas e agências:** Profissionais que utilizam PyMC-Marketing ou Meridian para modelagem de mix de marketing e buscam melhorar a especificação de seus modelos sem depender exclusivamente de experimentos de incrementalidade. O pipeline proposto é modular e integrável aos workflows existentes.

**Pesquisadores em inferência causal aplicada a marketing:** Acadêmicos e cientistas de dados interessados na interseção entre descoberta causal em séries temporais e modelagem Bayesiana. A pesquisa contribui com o primeiro benchmark comparativo de modelos Bayesianos de MMM com e sem calibração por descoberta causal.

**Gestores de marketing em mercados com dados limitados:** Decisores em organizações que operam com estratégias nacionais (poucos geos) e não dispõem de infraestrutura para geo-lift tests, mas necessitam de instrumentos quantitativos para alocação orçamentária fundamentada em causalidade.

---

# 4. Fundamentação Teórica

## 4.1 Marketing Mix Modeling: Origens, Conceitos e Evolução

[INSERIR CONTEÚDO EXISTENTE DO WIP — Seções 2.1 a 2.3]

O Marketing Mix Modeling (MMM) é uma técnica estatística que visa medir e quantificar o impacto de diferentes atividades e investimentos de marketing sobre indicadores de negócio como vendas, receita e volume de mercadoria (BERMAN, 2018; CHAN et al., 2017). Diferentemente de modelos de atribuição determinísticos, que rastreiam o percurso individual do cliente através dos pontos de contato, o MMM trabalha com dados agregados e utiliza técnicas econométricas para isolar o efeito de cada variável de marketing mantendo outras constantes.

### 4.1.1 Origens do Conceito de Marketing Mix

O conceito de Marketing Mix tem suas raízes na década de 1940, quando Culliton (1948, apud BORDEN, 1964) mencionou pela primeira vez a ideia de "mix de marketing". Neil Borden popularizou o termo em 1964, propondo um framework com 12 elementos componentes. A simplificação mais influente veio com McCarthy (1960, 1964), que consolidou os elementos nos famosos 4Ps: Product, Price, Place e Promotion, termos amplamente divulgados por Kotler (1967).

### 4.1.2 Modelos Econométricos Pioneiros

O desenvolvimento de modelos quantitativos para medir resposta de vendas à publicidade marca ponto de inflexão na história do marketing científico. O modelo de Vidale e Wolfe (1957) aplicou pesquisa operacional para modelar como gasto em publicidade se transforma em resposta de vendas, incorporando formalmente o conceito de decaimento do efeito publicitário. Nerlove e Arrow (1962) formalizaram a noção de publicidade como investimento de capital com depreciação temporal. Little (1970) desenvolveu o paradigma "Decision Calculus" com o modelo ADBUDG, representando transição de modelos puramente científicos para modelos práticos e acessíveis.

### 4.1.3 Evolução Contemporânea

[INSERIR CONTEÚDO DO WIP — Seções 2.4 a 2.8 sobre evolução técnica, deep learning, ferramentas open-source]

## 4.2 Efeitos Temporais e Não-Lineares em Marketing

### 4.2.1 Efeito Carryover (Adstock)

O efeito carryover refere-se ao impacto retardado dos investimentos em marketing ao longo do tempo. A modelagem adequada destes efeitos é crucial, pois sem considerá-los o MMM pode subestimar o impacto de canais com efeitos de longo prazo (TV, mídia impressa) e superestimar canais com efeitos imediatos (publicidade de busca).

As duas transformações dominantes são: geometric adstock, onde o efeito decai geometricamente com taxa α (utilizada por PyMC-Marketing), e delayed adstock com Weibull CDF (utilizada pelo Meridian), que permite modelar picos de efeito deslocados no tempo.

### 4.2.2 Efeito de Saturação (Diminishing Returns)

O efeito saturação refere-se aos rendimentos decrescentes associados ao aumento contínuo dos gastos em publicidade. As funções mais utilizadas são: Hill function (S-curve), parametrizada por ponto de inflexão e curvatura, e logistic saturation. O CausalMMM (GONG et al., 2024) utiliza S-curves com parâmetros aprendidos por redes neurais condicionadas a variáveis contextuais.

### 4.2.3 Não-Estacionariedade da Efetividade de Marketing

Dew, Padilla & Shchetkina (2024), em working paper do Marketing Science Institute, demonstram que efeitos não-lineares e time-varying são frequentemente não identificáveis separadamente em MMMs. Ng, Wang & Dai (2021), em trabalho no Uber apresentado no KDD/AdKDD Workshop, demonstram em escala industrial que parâmetros de efetividade de marketing são inerentemente variantes no tempo. Saggioro et al. (2020) introduzem RPCMCI para relações causais regime-dependentes. Coletivamente, esta literatura demonstra que calibrações estáticas — experimentais ou observacionais — degradam ao longo do tempo.

## 4.3 Frameworks Bayesianos para MMM

### 4.3.1 Fundamentos da Abordagem Bayesiana

Rossi & Allenby (2003) estabeleceram vantagens estruturais dos métodos Bayesianos para marketing: modularidade, incorporação de conhecimento prévio via priors, e quantificação rigorosa de incerteza via distribuições posteriores. A distribuição posterior Bayesiana captura simultaneamente estimativa central e intervalo de confiança condicional aos dados, permitindo comunicação sofisticada de incerteza para tomadores de decisão.

### 4.3.2 Calibração de Priors: Estado da Prática

Jin et al. (2017) propõem distribuições a priori derivadas de conhecimento de domínio sobre padrões de efetividade de mídia. Zhang et al. (2024), do Google, propõem reparametrização ROAS que permite especificação direta de priors a partir de "experiment results or the modeler's prior knowledge" — legitimando fontes não-experimentais. Korkames, Stanley & Stremersch (2025), na maior meta-análise de elasticidades publicitárias B2C (538 elasticidades, IJRM), argumentam que meta-análises "can serve as a viable and scalable alternative to expensive RCTs".

### 4.3.3 Limitações da Calibração Experimental

Gordon et al. (2019), analisando 15 experimentos em larga escala no Facebook (500M observações, Marketing Science), demonstram que evidência intervencionista é insubstituível para estimação de magnitude. No entanto, Lewis & Rao (2015, QJE) demonstram que experimentos informativos requerem mais de 10M pessoas-semana. Zhang et al. (2024) documentam três fontes de incompatibilidade entre experimentos e MMMs: (1) estimativas pontuais vs. períodos longos; (2) efeitos de canal único vs. interações multicanal; (3) potencial viés por confounders. Venkatraman et al. (2024) argumentam que calibração experimental pode inclusive piorar a acurácia do MMM em certos cenários.

## 4.4 Descoberta Causal em Séries Temporais

### 4.4.1 Causalidade Granger

A Causalidade Granger testa se valores passados de X fornecem informação significativa sobre valores futuros de Y, além do contido nos valores passados de Y. É baseada em previsibilidade temporal, não em intervenção (GRANGER, 1969). Kumar et al. (2024) propõem seu uso para seleção de variáveis em MMM. Limitações incluem: assume suficiência causal, linearidade na forma padrão, e estacionariedade.

### 4.4.2 PCMCI e Extensões

Runge (2020) introduz PCMCI+, que descobre relações causais contemporâneas e defasadas com conjuntos de condicionamento otimizados, melhorando poder de detecção sob autocorrelação — característica crítica de séries temporais de marketing. Gerhardus & Runge (2020) estendem para LPCMCI, que lida com confounders latentes. Assaad, Devijver & Gaussier (2022) publicam survey abrangente avaliando métodos constraint-based, score-based e Granger-based para séries temporais.

### 4.4.3 CD-NOTS: Descoberta Causal em Séries Temporais Não-Estacionárias

Sadeghi, Gopal & Fesanghary (2024) propõem o CD-NOTS, que estende o CD-NOD (HUANG et al., 2020) para séries temporais com relações defasadas. O algoritmo opera em quatro estágios: (1) adição de nó temporal indexado U_t para capturar não-estacionariedade; (2) descoberta de esqueleto causal via testes de independência condicional; (3) orientação de arestas via conhecimento prévio e V-structures; (4) orientação residual via independência de mudanças causais.

As suposições do CD-NOTS são: suficiência causal (confounders expressos como função suave do tempo), faithfulness, randomness, e consistência causal temporal. O CD-NOTS é não-paramétrico, capturando relações lineares e não-lineares, e a recomendação empírica do paper é ParCorr para N < 200 observações e KCIT/RCoT para N > 200 — compatível com dados típicos de MMM semanal.

**Implementação e seleção adaptativa do teste de independência condicional**

O presente trabalho implementa o CD-NOTS por meio da biblioteca tigramite (RUNGE, 2020), que fornece o PCMCI como motor de descoberta. O PCMCI aplica o teste MCI (Momentary Conditional Independence), que condiciona simultaneamente nos pais causais de ambas as variáveis testadas — mantendo os conjuntos de condicionamento pequenos independentemente do número de variáveis, ao contrário do algoritmo PC clássico, que sofre explosão combinatória em grafos com mais de 6–8 variáveis.

A seleção do teste de independência condicional é adaptativa ao número de séries temporais disponíveis. Para cenários multi-geo (dois ou mais geos), emprega-se ParCorr (correlação parcial com estatística Fisher-Z), 100 a 1000× mais rápido que métodos kernel. A replicação entre geos compensa em parte a pressuposição de linearidade: efeitos de adstock e saturação são aproximadamente lineares nos segmentos observacionais de dados de marketing semanal com N ≥ 100 observações por geo. Para cenários single-geo, emprega-se CMIknn (estimador k-NN de informação mútua condicional), não-paramétrico, capaz de capturar não-linearidades de adstock e saturação sem pressuposição de normalidade. O custo computacional O(N³) é aceitável com N ≤ 200 observações.

**Problema de múltiplos testes e correção de Benjamini-Hochberg**

O PCMCI realiza testes de independência condicional para cada par (i, j) de variáveis e cada defasagem τ ∈ {1, ..., τ_max}. Com 10 variáveis de descoberta (8 canais + 1 controle + y) e τ_max = 2, o número de testes simultâneos é da ordem de 10 × 10 × 2 = 200. A um nível de significância α = 0,05, esperam-se aproximadamente 10 falsos positivos sob a hipótese nula global — número inaceitável para um grafo com esparsidade esperada de 5–15 arestas verdadeiras.

Aplica-se a correção de Benjamini-Hochberg (BH; 1995) via `pcmci.get_corrected_pvalues(fdr_method="fdr_bh")`, que controla a taxa de falsas descobertas (FDR = E[FP/(FP+TP)]) ao nível nominal. O BH é preferível ao controle de FWER (Bonferroni, Holm) em descoberta causal porque aceita alguns falsos positivos em troca de maior poder de detecção — apropriado para grafos esparsos onde cada aresta verdadeira tem valor informativo (BENJAMINI; HOCHBERG, 1995). Os q-values BH-corrigidos são armazenados em `CausalGraph.edge_qvalues` e consumidos pelo módulo de calibração de priors, enquanto os p-values brutos são retidos em `CausalGraph.edge_pvalues` para fins de transparência e debug.

### 4.4.4 Métodos de Deep Learning para Descoberta Causal

Tank et al. (2022) propõem Neural Granger Causality com arquiteturas cMLP e cLSTM. Pamfil et al. (2020) introduzem DYNOTEARS, método score-based para redes Bayesianas dinâmicas. Löwe et al. (2022) propõem amortized causal discovery via inferência variacional. Gong et al. (2024) integram Graph VAE com Gumbel-Softmax para descoberta causal end-to-end em MMM. Estes métodos requerem volumes de dados significativamente maiores que métodos constraint-based.

## 4.5 Complementaridade entre Métodos Observacionais e Experimentais

Colnet et al. (2024, Statistical Science) demonstram que RCTs fornecem validade interna mas podem carecer de validade externa, enquanto dados observacionais fornecem representatividade mas sofrem confusão — a combinação melhora a inferência. Rosenman et al. (2023, Biometrics) provam formalmente que estimadores combinando dados observacionais e experimentais dominam estimadores puramente experimentais sob condições de amostra finita. Rosenman (2025, WIREs) documenta uma "explosão" de trabalho sobre combinação de efeitos causais experimentais e observacionais.

Athey, Chetty & Imbens (2025, NBER) desenvolvem o estimador ESC, demonstrando que experimentos fornecem a "chave de debiasing" enquanto dados observacionais fornecem escala. Ghassami et al. (2022) argumentam que nem dados experimentais nem observacionais isoladamente são suficientes para inferência causal de longo prazo.

A implicação para MMM é direta: a descoberta causal via CD-NOTS e a calibração experimental operam em dimensões complementares — CD-NOTS descobre a topologia (quais arestas existem), experimentos calibram a magnitude (qual o peso). Na ausência de experimentos, CD-NOTS oferece alternativa data-driven superior à especificação manual.

---

# 5. Soluções Existentes (Technology Road Map)

## 5.1 Estado da Prática: Frameworks Open-Source para MMM

### 5.1.1 Robyn (Meta)

Framework open-source utilizando ridge regression com constraints, otimização multi-objetivo e calibração via lift tests. Abordagem frequentista com elementos Bayesianos limitados.

### 5.1.2 PyMC-Marketing

Framework Bayesiano completo baseado em PyMC, com modelagem hierárquica geo-level, geometric adstock, Hill saturation, e posterior predictive checking. Permite customização granular de priors por canal. Utiliza MCMC (NUTS) com múltiplos samplers (nutpie, blackjax, numpyro).

### 5.1.3 Meridian (Google)

Framework Bayesiano geo-level com prior knowledge integration, Weibull adstock, e calibração via geo-experiments. Implementado em TensorFlow Probability. Inclui DAG causal fixo na documentação (canais → vendas, sem interações inter-canal).

### 5.1.4 Limitação Comum

Todos os frameworks assumem estrutura causal pré-definida pelo analista. A topologia do grafo é hardcoded: cada canal tem efeito independente sobre vendas. Relações inter-canal (TV → Search, Social → Organic) não são modeladas e não são descobertas automaticamente.

## 5.2 Estado da Arte: Descoberta Causal Aplicada a MMM

### 5.2.1 CausalMMM (Gong et al., 2024)

Primeiro método a integrar descoberta causal via Graph VAE com MMM. Utiliza Causal Relational Encoder (GNN com Gumbel-Softmax) e Marketing Response Decoder (GRU + S-curve). Demonstra 5,7-7,1% de melhoria em AUROC. **Limitação:** requer N ≥ 100 entidades para performance adequada; não quantifica incerteza; substitui (não complementa) frameworks Bayesianos.

### 5.2.2 DeepCausalMMM (Tirumala, 2025)

Combina GRU-based temporal modeling com NOTEARS DAG learning. **Limitação:** mesma dependência de volume de dados para descoberta de estrutura via redes neurais.

### 5.2.3 CDA — Causal-Driven Attribution (Filippou et al., 2025)

Aplica PCMCI a dados agregados e estima efeitos via SEM. **Limitação:** não integra com frameworks Bayesianos existentes; constrói pipeline separado.

### 5.2.4 CausalMTA (Yao et al., 2022)

Define SEM para atribuição multi-touch com decomposição de confounders estáticos e dinâmicos. Focado em dados de nível de usuário, não em dados agregados de MMM.

## 5.3 Lacuna Identificada

Nenhum trabalho publicado utiliza a saída de algoritmos de descoberta causal temporal como input para calibração de priors em modelos Bayesianos de MMM. Os trabalhos de descoberta causal (CausalMMM, DeepCausalMMM) propõem frameworks end-to-end de deep learning que substituem modelos Bayesianos. Os trabalhos de calibração Bayesiana (Jin et al., 2017; Zhang et al., 2024) utilizam experimentos ou meta-análises como fonte de priors. A ponte entre descoberta causal e calibração Bayesiana — usar grafos descobertos algoritmicamente para melhorar modelos já estabelecidos — é a lacuna que esta pesquisa endereça.

---

# 6. Solução Proposta

## 6.1 Pipeline: Descoberta Causal → Priors Estruturais → Modelagem Bayesiana

A contribuição metodológica central desta pesquisa é um pipeline de três estágios:

**Estágio 1 — Descoberta de Topologia (CD-NOTS):** A partir dos dados observacionais de marketing (séries temporais de investimento por canal e variável de resposta), CD-NOTS identifica a estrutura causal conjunta, incluindo: relações diretas canal → vendas, relações mediadas canal → canal → vendas, canais sem caminho causal para a variável de resposta, e relações não-estacionárias capturadas pelo nó temporal T. O CD-NOTS é particularmente adequado para dados de MMM por ser não-paramétrico e funcionar com 50-200 observações temporais (regime típico de MMM semanal).

**Estágio 2 — Tradução para Priors Estruturais via Empirical Bayes:** O grafo G = (V, E) descoberto é traduzido em modificações de priors utilizando o arcabouço de Empirical Bayes para testes múltiplos (STOREY, 2002; EFRON, 2010). O q-value BH-corrigido para cada aresta canal → vendas aproxima P(H₀ | dados), de modo que a probabilidade de inclusão posterior (PIP = 1 − q) aproxima P(efeito causal real | dados). Esta ponte conecta formalmente a descoberta causal com quantidade Bayesiana interpretável.

O desvio-padrão do prior de cada canal é ajustado via relaxação contínua do spike-and-slab (ISHWARAN; RAO, 2005):

σ_adj = σ_base × (MIN_SIGMA_RATIO + (1 − MIN_SIGMA_RATIO) × PIP)

Onde MIN_SIGMA_RATIO = 0,4 representa a largura mínima do componente spike (empiricamente validado: valores abaixo de 0,3 causam conflito prior-verossimilhança com R-hat > 1,8). Esta formulação é simétrica para canais diretos, mediados e excluídos — a diferença entre categorias está no q-value utilizado, não na direção da fórmula. Para adstock, o PIP modula o parâmetro b da distribuição Beta(1, b): canais causalmente confirmados recebem priors mais flexíveis (b menor, distribuição mais uniforme); canais excluídos recebem prior de decaimento rápido (b maior).

**Estágio 3 — Modelagem Bayesiana Informada:** O modelo Bayesiano (PyMC-Marketing ou Meridian) é estimado com os priors calibrados pelo grafo. Se resultados experimentais estiverem disponíveis, podem ser incorporados como camada adicional sobre a estrutura descoberta.

A viabilidade técnica é garantida pela modularidade dos frameworks: PyMC-Marketing permite ajuste de sigma no prior HalfNormal de beta_channel por canal; Meridian aceita parâmetros personalizados para beta_m (LogNormal) e alpha_m (Beta) por canal.

## 6.2 Design Experimental de 4 Braços

| Braço | Descoberta Causal | Modelagem | Priors |
|-------|-------------------|-----------|--------|
| 1 | Nenhuma (manual) | PyMC-Marketing | Padrão (spend-share) |
| 2 | Nenhuma (manual) | Meridian | Padrão (spend-share) |
| 3 | CD-NOTS | PyMC-Marketing | Calibrados pelo grafo |
| 4 | CD-NOTS | Meridian | Calibrados pelo grafo |

A comparação entre braços 1 vs 3 e 2 vs 4 isola o efeito da descoberta causal mantendo constante o paradigma de modelagem. Se os braços com CD-NOTS superarem consistentemente os baselines, demonstra-se que informação causal estrutural agrega valor independentemente do framework.

## 6.3 Métricas de Avaliação

### Métricas Preditivas
- R², MAPE, RMSE por geo e agregado
- Durbin-Watson (autocorrelação residual)

### Métricas de Atribuição (Ground Truth Recovery)
- Correlação e RMSE entre contribuições estimadas e verdadeiras por canal
- Recovery de ROAS por canal

### Métricas de Estrutura Causal
- Structural Hamming Distance (SHD) entre grafo descoberto e ground truth
- Precision, Recall e F1-score de arestas

### Métricas Bayesianas
- ESS (Effective Sample Size) mínimo
- R-hat convergence

---

# 7. Desenvolvimento (MVP)

## 7.1 Módulo de Descoberta Causal (`cdnots_discovery.py`)

O módulo implementa o pipeline de descoberta causal que recebe dados de marketing, detecta o formato de entrada, executa PCMCI por geo, aplica correção de FDR, e retorna um grafo de consenso com canais classificados em diretos, mediados e excluídos.

### 7.1.1 Estrutura de Dados: `CausalGraph`

O resultado da descoberta é encapsulado no dataclass imutável `CausalGraph`, com os seguintes campos principais:

- `adjacency_matrix` (n_vars × n_vars): matriz binária onde `adj[i,j] = 1` se a aresta i → j foi detectada no consenso
- `edge_pvalues`: p-values MCI brutos, retidos para transparência e debug
- `edge_qvalues`: q-values BH-corrigidos, consumidos pelo módulo de calibração de priors
- `variable_names`: sequência de canais + controles + "y" (ordem corresponde às linhas/colunas das matrizes)
- `direct_channels`, `mediated_channels`, `excluded_channels`: classificação de canais
- `endogenous_channels`, `endogenous_r2`: canais confundidos por variável de controle e R² correspondente
- `ci_test_used`: "parcorr" ou "kci", para auditabilidade dos resultados

A distinção entre `edge_pvalues` e `edge_qvalues` é central: a versão anterior do módulo utilizava p-values brutos tanto para decisão de inclusão de arestas quanto para calibração de priors, o que violava o controle de múltiplos testes e produzia FDR = 73% no preset `causal_business`.

### 7.1.2 Resolução de Formato: `_resolve_data_format`

A função `prepare_dataset_for_modeling` do framework de benchmark retorna um DataFrame com colunas `"time"` e `"geo"` no índice regular — não em MultiIndex. O módulo de descoberta espera os dados separados por geo. Sem conversão explícita, todos os 4 geos × 156 semanas = 624 linhas eram concatenados em uma única pseudo-série "nacional", introduzindo dependências temporais artificiais nas fronteiras entre geos e causando FDR elevado.

A função `_resolve_data_format` resolve este problema detectando automaticamente o formato de entrada: se o índice já é MultiIndex, retorna sem alteração; se houver colunas `"time"`/`"date"` e `"geo"`, constrói o MultiIndex correspondente, normalizando o nome do nível de tempo para `"date"` (necessário porque o código downstream usa `get_level_values("date")` independentemente do nome original). Na ausência de coluna geo, retorna o DataFrame original, preservando o comportamento nacional para presets single-geo.

### 7.1.3 Fluxo Principal: `discover_graph`

A função `discover_graph` orquestra a descoberta em seis etapas:

1. **Resolução de formato:** converte o DataFrame de entrada para MultiIndex se necessário
2. **Seleção de CI test:** multi-geo → parcorr; single-geo → CMIknn (lógica descrita na Seção 4.4.3)
3. **Loop por geo:** para cada geo (máximo de 5, amostrados aleatoriamente se mais disponíveis), executa `_discover_single_geo`, que retorna `(adj, pval, qval)` para aquele geo
4. **Consenso por maioria de votos:** uma aresta é incluída no grafo final se detectada em ao menos 50% dos geos; p-values e q-values do grafo de consenso são médias condicionais sobre os geos que detectaram cada aresta
5. **Restrições estruturais:** y não causa outros canais (direcionalidade temporal); controles são tratados como exógenos (canais não causam controles por hipótese)
6. **Classificação e endogeneidade:** canais diretos (aresta ch → y), mediados (caminho ch → ... → y via BFS com profundidade ≤ 3), excluídos (sem caminho); canais causados por controles são marcados endógenos com R² calculado como penalidade

### 7.1.4 Descoberta por Geo: `_pcmci_discovery`

Para cada geo, os dados são padronizados (StandardScaler, média zero e desvio 1) e passados ao PCMCI:

- Configuração: `tau_min=1`, `tau_max=max_lag` — exclui arestas contemporâneas, que são ambíguas em dados de marketing com granularidade semanal
- `p_matrix[i, j, τ]` contém o p-value MCI de X_i(t−τ) → X_j(t)
- `get_corrected_pvalues(fdr_method="fdr_bh")` aplica BH sobre todos os pares e defasagens
- Decisão de inclusão: `adj[i,j] = 1` se `min(q_matrix[i,j,1:τ_max+1]) < α`

**Fallbacks:** na ausência de tigramite, o módulo recorre ao algoritmo PC com augmentação temporal via causal-learn; na ausência de ambos, usa Granger pairwise via statsmodels. Nos fallbacks, `qval = pval` (sem correção FDR disponível), comportamento documentado com aviso explícito ao usuário.

## 7.2 Módulo de Calibração de Priors (`cdnots_model_builder.py`)

### 7.2.1 Fundamentação: Empirical Bayes e Probabilidade de Inclusão Posterior

Sob o modelo Empirical Bayes de Storey (2002), o q-value de Benjamini-Hochberg satisfaz:

q_i ≈ P(H₀ | dados_i)

onde H₀ é a hipótese nula de ausência de aresta causal. Portanto, a probabilidade de inclusão posterior (PIP) — probabilidade de que o efeito causal seja real dado os dados — é:

**PIP_i = 1 − q_i**

Esta identidade conecta formalmente a descoberta causal com quantidade Bayesiana interpretável (EFRON, 2010, Cap. 5). Um canal com q = 0,01 tem PIP = 0,99: há 99% de probabilidade de que o efeito causal seja real. Um canal excluído com q = 0,95 tem PIP = 0,05: há apenas 5% de probabilidade de efeito real — o prior deve regularizar fortemente para zero.

### 7.2.2 Relaxação Contínua do Spike-and-Slab

O prior ideal para cada coeficiente de canal β é a mistura discreta:

β ~ PIP × HalfNormal(σ_base) + (1 − PIP) × δ(0)

onde δ(0) é a massa pontual em zero. Esta especificação é intratável em MCMC porque requer variáveis latentes binárias, causando convergência lenta. Emprega-se a **relaxação contínua do spike-and-slab** (ISHWARAN; RAO, 2005), que substitui δ(0) por uma HalfNormal de largura mínima, interpolando linearmente entre os dois extremos:

**σ_adj = σ_base × (MIN_SIGMA_RATIO + (1 − MIN_SIGMA_RATIO) × PIP)**

### 7.2.3 Derivação das Constantes

**MIN_SIGMA_RATIO = 0,4** é a largura mínima do componente spike. Empiricamente, valores abaixo de 0,3 causam conflito prior-verossimilhança, manifestado como R-hat > 1,8 no diagnóstico MCMC. O valor 0,4 é o mínimo que preserva tratabilidade do sampler, determinado empiricamente durante o desenvolvimento. Este parâmetro corresponde ao `FLOOR = 0,4` da versão anterior do módulo, renomeado para `MIN_SIGMA_RATIO` para tornar seu papel teórico explícito.

**(1 − MIN_SIGMA_RATIO) = 0,6** não é uma escolha independente — é consequência algébrica de MIN_SIGMA_RATIO = 0,4 e da exigência de interpolação linear entre os extremos (spike = 0,4·σ_base quando PIP = 0; slab = σ_base quando PIP = 1). A fórmula não tem parâmetros livres além de MIN_SIGMA_RATIO.

A tabela abaixo ilustra os multiplicadores para valores típicos de q-value:

| Tipo de canal | q-value | PIP = 1−q | σ_adj / σ_base |
|---------------|---------|-----------|----------------|
| Direto, evidência forte | 0,01 | 0,99 | ≈ 1,00 |
| Direto, moderado | 0,20 | 0,80 | 0,88 |
| Mediado, borderline | 0,50 | 0,50 | 0,70 |
| Excluído, evidência fraca | 0,20 | 0,80 | 0,88 (dados decidem) |
| Excluído, confiante | 0,90 | 0,10 | 0,46 |
| Excluído, muito confiante | 0,99 | 0,01 | ≈ 0,40 |

Um canal excluído com q = 0,20 (evidência fraca de exclusão) recebe multiplicador 0,88 — quase sem regularização — porque a incerteza é alta e o modelo deve deferir aos dados. Um canal excluído com q = 0,99 recebe multiplicador 0,40 (máxima regularização possível). Esta assimetria é conceitualmente correta: o prior não deve punir um canal quando a evidência contra ele é fraca.

### 7.2.4 Aplicação por Categoria de Canal

A fórmula é idêntica para todas as categorias; o que difere é o q-value utilizado:

- **Canais diretos:** q-value da aresta ch → y em `edge_qvalues[ch_idx, y_idx]`
- **Canais mediados:** q-value do caminho mais confiante até y, calculado via BFS de elo-mais-fraco: entre todos os caminhos ch → ... → y com profundidade ≤ 3, escolhe-se o caminho com menor q-value máximo (menor incerteza no elo mais fraco do caminho)
- **Canais excluídos:** q-value da aresta ch → y em `edge_qvalues[ch_idx, y_idx]`, que é alto por definição (aresta não detectada), resultando em PIP baixo e máxima regularização

### 7.2.5 Prior de Adstock

O prior de decaimento geométrico é parametrizado como Beta(1, α_b). Valores altos de α_b concentram a distribuição próxima de zero (decaimento rápido). O PIP modula α_b linearmente:

α_b = clip(BASE_B − (BASE_B − MIN_B) × PIP, MIN_B, 5,0)

Com BASE_B = 3,0 e MIN_B = 1,0: canal com PIP ≈ 1 → α_b ≈ 1,0 (Beta(1,1) = Uniforme, adstock muito flexível); canal com PIP ≈ 0 → α_b ≈ 3,0 (Beta(1,3), prior de decaimento rápido).

### 7.2.6 Penalidade de Endogeneidade

Para canais detectados como confundidos por variável de controle (controle → canal no grafo), aplica-se penalidade proporcional ao R² (variância do canal explicada pelo controle):

tolerance = max(MIN_SIGMA_RATIO, 1 − R²)
multiplier_final = max(MIN_SIGMA_RATIO, multiplier_PIP × tolerance)

O R² é calculado como quadrado da correlação entre controle e canal nos dados observados, fornecendo medida data-driven do grau de endogeneidade. A penalidade é composta com o multiplicador baseado em PIP, sempre respeitando o piso MIN_SIGMA_RATIO.

### 7.2.7 Integração com PyMC-Marketing e Meridian

Para **PyMC-Marketing**, os multiplicadores são aplicados ao desvio-padrão do prior HalfNormal do coeficiente de canal (`sigma` do prior de `beta_channel`). Para **Meridian**, são aplicados ao parâmetro σ do prior LogNormal de `beta_m`. Em ambos os frameworks, `_compute_adstock_params` ajusta os parâmetros da distribuição Beta do prior de adstock (`alpha` em PyMC-Marketing, `alpha_m` no Meridian).

## 7.3 Geração do Dataset com Estrutura Causal Conhecida

### 7.3.1 Motivação

A avaliação empírica do pipeline proposto requer dados sintéticos com ground truth conhecido: quais canais realmente afetam as vendas, quais arestas causais inter-canal existem, e quais os verdadeiros parâmetros de adstock e saturação. O framework de benchmark `mmm_param_recovery` fornece um gerador de dados parametrizado que implementa exatamente esta estrutura, permitindo calcular métricas de recuperação (Precision, Recall, FDR, SHD) com referência ao ground truth.

### 7.3.2 Preset `causal_business`

O preset `causal_business` foi desenvolvido especificamente para o benchmark CD-NOTS, simulando uma empresa com presença em 4 regiões geográficas e um funnel de marketing realista. Suas características são:

| Dimensão | Valor |
|----------|-------|
| Períodos | 156 semanas (3 anos) |
| Geos | 4 (geo_a, geo_b, geo_c, geo_d) |
| Canais totais | 8 |
| Canais com efeito real | 6 (Search-Ads, Brand-Search, TV, Video, Social-Media, Display-Ads) |
| Canais ghost | 2 (Ghost-A, Ghost-B: `base_effectiveness = 0`) |
| Controles | 1 (preço) |
| Arestas causais inter-canal | 3 |

Os canais ghost são incluídos para testar a capacidade do pipeline de suprimir canais sem efeito real — caso em que o prior calibrado deve apresentar multiplicador próximo de MIN_SIGMA_RATIO = 0,4.

### 7.3.3 Estrutura de Arestas Causais: `CausalEdgeConfig`

As arestas causais inter-canal são especificadas via o dataclass `CausalEdgeConfig`:

| Campo | Descrição |
|-------|-----------|
| `source_channel` | Canal de origem do spillover |
| `target_channel` | Canal que recebe o spillover |
| `effect_size` | Fração do spend adstockado do source que se propaga ao target (0 < e ≤ 1) |
| `lag` | Defasagem em períodos antes do efeito se manifestar |
| `decay` | Taxa de decaimento geométrico do adstock do source antes do spillover |

As três arestas do preset `causal_business` modelam o funnel upper-funnel → lower-funnel:

| Aresta | lag | effect_size | decay | Interpretação |
|--------|-----|-------------|-------|---------------|
| TV → Search-Ads | 2 semanas | 0,20 | 0,50 | Publicidade de branding em TV aumenta a busca paga 2 semanas depois |
| Social-Media → Brand-Search | 1 semana | 0,15 | 0,40 | Engajamento social impulsiona busca por marca na semana seguinte |
| Video → Social-Media | 1 semana | 0,10 | 0,30 | Vídeo online aumenta engajamento em redes sociais |

### 7.3.4 Implementação do Spillover Causal em `_generate_channel_spend_data`

A geração de dados opera em duas fases sequenciais para cada geo:

**Fase 1 — Spend base independente:** cada canal gera sua série temporal de investimento conforme seu padrão configurado (`linear_trend`, `seasonal`, `on_off`), com variações regionais determinísticas por geo (controladas por seed). Os canais são independentes entre si nesta fase.

**Fase 2 — Spillover causal:** para cada aresta `CausalEdgeConfig`, o spend do canal fonte é adstockado geometricamente com taxa `decay` e então adicionado ao spend do canal alvo com a defasagem `lag`:

```
adstocked[t] = source_spend[t] + decay × adstocked[t−1]
spillover[t]  = effect_size × adstocked[t − lag]   (para t ≥ lag)
channel_spends[target] += spillover
```

O `effect_size` representa a fração do spend adstockado do canal fonte que se manifesta como spend adicional no canal alvo. Para TV → Search-Ads com `effect_size=0,20` e `lag=2`: cada unidade de spend de TV gera, após adstock com decaimento 0,5, um aumento de 20% desse valor no spend de Search-Ads dois períodos depois. As arestas são processadas em ordem de definição; efeitos mediados (Video → Social-Media → Brand-Search) emergem composicionalmente ao longo dos períodos.

### 7.3.5 Ground Truth para Avaliação

A função `_build_causal_ground_truth` constrói a matriz de adjacência verdadeira a partir da configuração, registrando: (a) arestas inter-canal definidas em `causal_edges`; (b) arestas canal → y para canais com `base_effectiveness > 0`; (c) ausência de arestas para canais ghost. Esta matriz é comparada com o grafo descoberto pelo CD-NOTS para calcular as métricas estruturais.

## 7.4 Framework de Benchmark

O módulo `cdnots_fitter.py` orquestra a execução completa dos Braços 3 e 4 do experimento:

1. Invoca `discover_graph` para obter o `CausalGraph` a partir dos dados de treinamento
2. Repassa o grafo para `build_pymc_model_with_graph` (Braço 3) ou `build_meridian_model_with_graph` (Braço 4)
3. Executa MCMC com os priors calibrados pelos multiplicadores Empirical Bayes
4. Coleta métricas de convergência (R-hat, ESS) e de atribuição (ROAS por canal, contribuições)

A integração com o pipeline de benchmark existente é realizada via `CDNOTS_INTEGRATION.py`, que contém os patches para `run_benchmark.py` do repositório `mmm_param_recovery`. Os patches adicionam os Braços 3 e 4 ao loop de benchmark sem alterar os Braços 1 e 2 (baselines PyMC e Meridian), garantindo que todas as comparações sejam feitas com dados idênticos e mesmos hiperparâmetros MCMC.

---

# 8. Validação

## 8.1 Dados Sintéticos com Ground Truth Conhecido

O experimento de validação principal utiliza o preset `causal_business` (descrito em detalhes na Seção 7.3) como caso primário de avaliação. Este preset foi escolhido por três razões: possui estrutura causal inter-canal explícita e conhecida, permitindo avaliação precisa do módulo de descoberta; é multi-geo (4 geos), exercitando o caminho de código corrigido para detecção de múltiplas séries temporais; e inclui canais ghost com effectiveness zero, testando a capacidade do pipeline de suprimir canais irrelevantes.

O pipeline foi executado com o preset na configuração anterior ao pipeline corrigido para estabelecer uma linha de base quantitativa dos problemas identificados:

| Métrica | Valor baseline (pré-correção) |
|---------|-------------------------------|
| Precision | 0,267 |
| Recall | 0,444 |
| F1 | 0,333 |
| FDR | 0,733 |
| SHD | 16 |
| Verdadeiros Positivos (TP) | 4 |
| Falsos Positivos (FP) | 11 |
| Falsos Negativos (FN) | 5 |

O FDR = 73% confirma o diagnóstico: 11 falsos positivos para apenas 4 verdadeiros positivos. O algoritmo detectava arestas causais inexistentes em massa porque operava sobre a série concatenada de todos os 4 geos como uma única série nacional de 624 observações, gerando dependências temporais artificiais nas fronteiras entre geos. A seleção incorreta de CMIknn (ativada para N = 1 geo efetivo) em vez de parcorr agravava o problema, pois CMIknn tem maior taxa de falsos positivos para séries longas.

## 8.2 Protocolo de Validação

Para o preset `causal_business`, o protocolo de validação segue cinco etapas sequenciais:

1. **Estrutura causal:** executar `discover_graph` com o pipeline corrigido → comparar grafo com ground truth via `evaluate_causal_structure` → calcular SHD, Precision, Recall, F1, FDR. Critérios de sucesso: FDR ≤ 0,40, Precision ≥ 0,60, `ci_test_used == "parcorr"`.

2. **Calibração de priors:** inspecionar a saída de `_log_adjustments` → canais com q ≈ 0,01 devem ter multiplicador ≈ 1,0; canais ghost devem ter multiplicador próximo de 0,40. Verificar diferenciação significativa entre categorias direct/excluded.

3. **Convergência MCMC:** executar Braços 3 (PyMC+CD-NOTS) e 4 (Meridian+CD-NOTS) → verificar R-hat < 1,05 para todos os parâmetros. O limiar 1,05 é mais restritivo que o convencional 1,1 para garantir ausência de conflito prior-verossimilhança — diagnóstico que motivou o redesign do módulo de calibração.

4. **Recuperação de atribuição:** comparar contribuições estimadas com as contribuições verdadeiras por canal → calcular correlação de Pearson e RMSE do ROAS por canal. Verificar se canais ghost têm ROAS estimado próximo de zero em todos os braços.

5. **Comparação dos 4 braços:** quantificar o ganho marginal da calibração CD-NOTS comparando Braço 1 vs. Braço 3 (PyMC) e Braço 2 vs. Braço 4 (Meridian) em todas as métricas preditivas e de atribuição.

---

# 9. Resultados

## 9.1 Resultados do Benchmark PyMC vs Meridian (Baseline)

[INSERIR resultados já disponíveis do repo mmm-param-recovery — braços 1 e 2]

## 9.2 Resultados da Descoberta Causal via CD-NOTS

[PLACEHOLDER — a ser preenchido após execução]

## 9.3 Resultados Comparativos: Baseline vs Graph-Informed

[PLACEHOLDER — a ser preenchido após execução]

---

# 10. Impactos

## 10.1 Impacto Acadêmico

Esta pesquisa contribui com a primeira proposta e avaliação empírica de integração entre descoberta causal algorítmica e calibração de priors Bayesianos em MMM. A lacuna entre as linhas de pesquisa em causal discovery (CausalMMM, CD-NOTS) e Bayesian MMM (PyMC, Meridian) é formalmente endereçada.

## 10.2 Impacto Prático

O pipeline proposto é modular e integrável aos workflows existentes de empresas que já utilizam PyMC-Marketing ou Meridian. Reduz a dependência de experimentos de incrementalidade para calibração, oferecendo alternativa data-driven acessível a organizações sem infraestrutura experimental.

## 10.3 Impacto para o Mercado Brasileiro

A hipótese de que abordagens de redes neurais para descoberta causal em MMM (CausalMMM, DeepCausalMMM) podem não ser adotadas no mercado nacional — devido à predominância de estratégias de mídia nacionais e consequente escassez de datapoints geo-level — é discutida e validada. A abordagem proposta (CD-NOTS + Bayesiano) constitui alternativa viável para este contexto.

---

# 11. Plano de Exploração da Tecnologia

## 11.1 Technology Readiness Level (TRL)

O artefato encontra-se em **TRL 4** (validação em ambiente de laboratório com dados sintéticos). Próximos passos para avanço:
- TRL 5: validação com dados reais anonimizados de parceiro
- TRL 6: demonstração em ambiente operacional
- TRL 7: protótipo em ambiente operacional

## 11.2 Ativos de Propriedade Intelectual

- Repositórios open-source (MIT License):
  - https://github.com/rafaennes/causalmmm_cdnots
  - Framework de benchmark mmm-param-recovery

## 11.3 Ecossistema

- Integração com comunidades open-source de PyMC-Marketing e Meridian
- Potencial contribuição como módulo/plugin para os frameworks existentes

---

# 12. Artefatos Tecnológicos Gerados

| Artefato | Descrição | Disponibilização |
|----------|-----------|------------------|
| cdnots_graph_discovery.py | Módulo de descoberta causal CD-NOTS para dados de MMM | GitHub: rafaennes/causalmmm_cdnots |
| graph_informed_model_builder.py | Tradução grafo → priors para PyMC e Meridian | GitHub: rafaennes/causalmmm_cdnots |
| benchmark_with_cdnots.py | Orquestração do benchmark de 4 braços | GitHub: rafaennes/causalmmm_cdnots |
| Dataset benchmark | Dados sintéticos com ground truth | GitHub: mmm-param-recovery |

---

# 13. Publicações

## 13.1 Plano de Publicação

| Veículo-alvo | Tipo | Título provisório | Status |
|--------------|------|-------------------|--------|
| KDD ou WSDM Workshop | Short paper | "Causal Graph Priors for Bayesian Marketing Mix Models" | Em preparação |
| International Journal of Research in Marketing | Artigo | "Integrating Causal Discovery with Bayesian MMM: A Comparative Study" | Planejado |

---

# 14. Referências

[INSERIR referências consolidadas — as 32 da seção v2 + referências existentes do WIP]

ASSAAD, C.K.; DEVIJVER, E.; GAUSSIER, É. Survey and Evaluation of Causal Discovery Methods for Time Series. Journal of Artificial Intelligence Research, v. 73, p. 767-819, 2022.

BENJAMINI, Y.; HOCHBERG, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society. Series B (Methodological), v. 57, n. 1, p. 289-300, 1995.

ATHEY, S.; CHETTY, R.; IMBENS, G.W. The Experimental Selection Correction Estimator. NBER Working Paper 33817, 2025.

BERMAN, R. Beyond the Last Touch: Attribution in Online Advertising. Marketing Science, v. 37, n. 5, 2018.

BORDEN, N.H. The Concept of the Marketing Mix. Journal of Advertising Research, v. 4, n. 2, p. 2-7, 1964.

CHAN, D. et al. Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. Google Research, 2017.

CHEN, A. et al. Bias Correction for Paid Search in Media Mix Modeling. arXiv:1807.03292, 2018.

CHEN, H. et al. Hierarchical Marketing Mix Models with Sign Constraints. Journal of Applied Statistics, v. 48, n. 13-15, p. 2944-2960, 2021.

COLNET, B. et al. Causal Inference Methods for Combining Randomized Trials and Observational Studies. Statistical Science, v. 39, n. 1, p. 165-191, 2024.

DEKIMPE, M.G.; HANSSENS, D.M. The Persistence of Marketing Effects on Sales. Marketing Science, v. 14, n. 1, p. 1-21, 1995.

DEW, R.; PADILLA, N.; SHCHETKINA, A. Your MMM is Broken. MSI Working Paper 24-144, arXiv:2408.07678, 2024.

EFRON, B. Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction. Cambridge University Press, 2010.

FILIPPOU, G. et al. Causal-Driven Attribution (CDA). arXiv:2512.21211, 2025.

GERHARDUS, A.; RUNGE, J. High-Recall Causal Discovery for Autocorrelated Time Series with Latent Confounders. NeurIPS, p. 12615-12625, 2020.

GHASSAMI, A. et al. Combining Experimental and Observational Data for Long-Term Causal Effects. arXiv:2201.10743, 2022.

GONG, C. et al. CausalMMM: Learning Causal Structure for Marketing Mix Modeling. WSDM '24, p. 238-246, 2024.

GORDON, B.R.; MOAKLER, R.; ZETTELMEYER, F. Close Enough? Marketing Science, v. 42, n. 4, p. 768-793, 2023.

GORDON, B.R. et al. A Comparison of Approaches to Advertising Measurement. Marketing Science, v. 38, n. 2, p. 193-225, 2019.

GRANGER, C.W.J. Investigating Causal Relations by Econometric Models and Cross-spectral Methods. Econometrica, v. 37, n. 3, p. 424-438, 1969.

HANSSENS, D.M. Using Return on Marketing Investment Effectively. Journal of Marketing Research, 2024.

HANSSENS, D.M.; PAUWELS, K.H. Demonstrating the Value of Marketing. Journal of Marketing, v. 80, n. 6, p. 173-190, 2016.

HUANG, B. et al. Causal Discovery from Heterogeneous/Nonstationary Data. JMLR, v. 21, n. 89, p. 1-53, 2020.

ISHWARAN, H.; RAO, J.S. Spike and slab variable selection: frequentist and Bayesian strategies. Annals of Statistics, v. 33, n. 2, p. 730-773, 2005.

JIN, Y. et al. Bayesian Methods for Media Mix Modeling. Google Research, 2017.

KORKAMES, J.; STANLEY, T.D.; STREMERSCH, S. Meta-Analysis of Advertising Effectiveness. IJRM, v. 42, n. 4B, p. 1264-1283, 2025.

KUMAR, R. et al. Granger Causality as Feature Selection in MMM. TechRxiv, 2024.

LEWIS, R.A.; RAO, J.M. The Unfavorable Economics of Measuring Advertising Returns. QJE, v. 130, n. 4, p. 1941-1973, 2015.

LÖWE, S. et al. Amortized Causal Discovery. CLeaR, PMLR 177, p. 509-525, 2022.

MARCINKEVIČS, R.; VOGT, J.E. Interpretable Models for Granger Causality. ICLR, 2021.

MCCARTHY, E.J. Basic Marketing: A Managerial Approach. Irwin, 1960.

MORGAN, N.A. et al. Marketing Performance Assessment and Accountability. IJRM, v. 39, n. 2, p. 462-481, 2022.

NG, E.; WANG, Z.; DAI, A. Bayesian Time Varying Coefficient Model for MMM. KDD/AdKDD, arXiv:2106.03322, 2021.

PAMFIL, R. et al. DYNOTEARS: Structure Learning from Time-Series Data. AISTATS, PMLR 108, p. 1595-1605, 2020.

PEARL, J. Causality: Models, Reasoning, and Inference. 2. ed. Cambridge University Press, 2009.

ROSENMAN, E.T.R. Methods for Combining Observational and Experimental Causal Estimates. WIREs, v. 17, e70027, 2025.

STOREY, J.D. A direct approach to false discovery rates. Journal of the Royal Statistical Society. Series B (Statistical Methodology), v. 64, n. 3, p. 479-498, 2002.

ROSENMAN, E.T.R. et al. Combining Datasets Using Shrinkage Estimators. Biometrics, v. 79, n. 4, p. 2961-2973, 2023.

ROSSI, P.E.; ALLENBY, G.M. Bayesian Statistics and Marketing. Marketing Science, v. 22, n. 3, p. 304-328, 2003.

RUNGE, J. Discovering Contemporaneous and Lagged Causal Relations. UAI, PMLR 124, p. 1388-1397, 2020.

SADEGHI, A.; GOPAL, A.; FESANGHARY, M. Causal Discovery in Financial Markets. IJDSA, 2024.

SAGGIORO, E. et al. Reconstructing Regime-Dependent Causal Relationships. Chaos, v. 30, n. 11, 113115, 2020.

TANK, A. et al. Neural Granger Causality. IEEE TPAMI, v. 44, n. 8, p. 4267-4279, 2022.

TIRUMALA, A.P. DeepCausalMMM. arXiv:2510.13087, 2025.

VAN DER WEIDE, W. et al. Unified Marketing Measurement. Think with Google, 2020.

VENKATRAMAN, R. et al. Why You Should Not Calibrate MMM Through Experiments. TechRxiv, 2024.

WANG, Y. et al. A Hierarchical Bayesian Approach to Improve Media Mix Models. Google Research, 2017.

YAO, D. et al. CausalMTA. KDD '22, p. 4342-4352, 2022.

ZHANG, Y. et al. Media Mix Model Calibration With Bayesian Priors. Google LLC, 2024.
