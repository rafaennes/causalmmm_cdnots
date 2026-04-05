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
   - 7.2 Módulo de Tradução Grafo → Priors
   - 7.3 Integração com PyMC-Marketing e Meridian
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

**Estágio 2 — Tradução para Priors Estruturais:** O grafo G = (V, E) descoberto é traduzido em modificações concretas na especificação do modelo Bayesiano:

- Canais com aresta direta forte para vendas: priors com variância maior (menos regularização, deixa os dados informar a magnitude)
- Canais sem caminho causal identificado: priors com variância muito pequena (forte regularização para zero — seleção de variáveis soft)
- Canais com efeitos mediados: priors com variância intermediária
- Para adstock: canais com efeitos defasados no CD-NOTS recebem priors de decay mais lentos; canais com efeitos apenas contemporâneos recebem priors de decay rápido

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

## 7.1 Módulo de Descoberta Causal (CD-NOTS)

[Descrição técnica do cdnots_graph_discovery.py — já implementado]

O módulo recebe dados no formato MultiIndex (date, geo), executa CD-NOTS por geo com fallback para causal-learn PC algorithm e Granger causality, constrói grafo de consenso via votação majoritária, e retorna CausalGraphResult com matriz de adjacência, forças de arestas, canais diretos/excluídos/mediados.

## 7.2 Módulo de Tradução Grafo → Priors

[Descrição técnica do graph_informed_model_builder.py — já implementado]

O módulo traduz CausalGraphResult em modificações de priors para PyMC-Marketing (sigma do HalfNormal × 1.2 para diretos, × 0.8 para mediados, × 0.1 para excluídos) e Meridian (sigma do LogNormal + ajuste de alpha_m Beta para adstock).

## 7.3 Integração com PyMC-Marketing e Meridian

[Descrição de fit_pymc_with_cdnots e fit_meridian_with_cdnots — já implementados]

## 7.4 Framework de Benchmark

[Descrição do benchmark_with_cdnots.py e integração com run_benchmark.py do repo mmm-param-recovery]

Repositório: https://github.com/rafaennes/causalmmm_cdnots

---

# 8. Validação

## 8.1 Dados Sintéticos com Ground Truth Conhecido

Utilização do data_generator do framework mmm-param-recovery para gerar datasets com:
- Estrutura causal conhecida (quais canais realmente afetam vendas)
- Parâmetros de adstock e saturação conhecidos
- Contribuições verdadeiras por canal
- ROAS verdadeiro por canal e geo

Presets: small_business, medium_business, large_business, growing_business.

## 8.2 Protocolo de Validação

Para cada dataset:
1. Rodar CD-NOTS → comparar grafo descoberto com ground truth (SHD, F1)
2. Estimar PyMC baseline e PyMC + CD-NOTS com mesmos hiperparâmetros MCMC
3. Estimar Meridian baseline e Meridian + CD-NOTS com mesmos hiperparâmetros
4. Comparar métricas preditivas e de atribuição entre baseline e graph-informed

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

ATHEY, S.; CHETTY, R.; IMBENS, G.W. The Experimental Selection Correction Estimator. NBER Working Paper 33817, 2025.

BERMAN, R. Beyond the Last Touch: Attribution in Online Advertising. Marketing Science, v. 37, n. 5, 2018.

BORDEN, N.H. The Concept of the Marketing Mix. Journal of Advertising Research, v. 4, n. 2, p. 2-7, 1964.

CHAN, D. et al. Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. Google Research, 2017.

CHEN, A. et al. Bias Correction for Paid Search in Media Mix Modeling. arXiv:1807.03292, 2018.

CHEN, H. et al. Hierarchical Marketing Mix Models with Sign Constraints. Journal of Applied Statistics, v. 48, n. 13-15, p. 2944-2960, 2021.

COLNET, B. et al. Causal Inference Methods for Combining Randomized Trials and Observational Studies. Statistical Science, v. 39, n. 1, p. 165-191, 2024.

DEKIMPE, M.G.; HANSSENS, D.M. The Persistence of Marketing Effects on Sales. Marketing Science, v. 14, n. 1, p. 1-21, 1995.

DEW, R.; PADILLA, N.; SHCHETKINA, A. Your MMM is Broken. MSI Working Paper 24-144, arXiv:2408.07678, 2024.

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
