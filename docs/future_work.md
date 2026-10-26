# Implementações Futuras

## 1. Substituir `parcorr` por `RobustParCorr` como teste CI padrão

**Arquivo:** `cdnots/cdnots_discovery.py`

### Motivação

O teste de independência condicional padrão (`parcorr`) assume linearidade na relação entre variáveis. Dados de MMM violam essa suposição sistematicamente: o spend de cada canal passa por adstock geométrico e função de saturação Hill antes de afetar y. Ambas as transformações são **monotônicas**, mas não lineares.

Resultado observado: nos presets `causal_business` e `causal_large`, o `parcorr` classifica a maioria dos canais com efeito real como `excluded` (F1 ≈ 0.0–0.22), comprimindo os priors desses canais para 40% (MIN_SIGMA_RATIO) e prejudicando a convergência MCMC (ESS mín caiu de 121 → 58 no `causal_business`).

### Alternativa: `RobustParCorr`

`RobustParCorr` é um teste de correlação parcial baseado em **postos** (modelo não-paranormal / copula):

1. Transforma as marginais de cada variável para normal padrão via ranqueamento
2. Aplica parcorr nos resíduos ranqueados

Como adstock e Hill são transformações monotônicas, a correlação de postos entre `spend_i` e `y` é bem mais alta do que a correlação de Pearson. O `RobustParCorr` captura essa dependência sem precisar de estimativas explícitas dos parâmetros de adstock ou saturação.

**Disponível no ambiente atual** via `tigramite.independence_tests.robust_parcorr.RobustParCorr` — sem dependências extras.

### Implementação

Em `cdnots/cdnots_discovery.py`:

1. **Ampliar a validação de `ci_test`** para aceitar `"robust_parcorr"`:

```python
# linha ~187 — adicionar "robust_parcorr" às opções aceitas
if ci_test not in ("parcorr", "kci", "robust_parcorr"):
    raise ValueError(...)
```

2. **Atualizar o `auto` mode** para usar `robust_parcorr` em vez de `parcorr`:

```python
if ci_test == "auto":
    ci_test = "kci" if len(geos) == 1 else "robust_parcorr"
    # antes: ci_test = "kci" if len(geos) == 1 else "parcorr"
```

3. **Instanciar `RobustParCorr` em `_pcmci_discovery`**:

```python
# em _pcmci_discovery(), após instanciar ParCorr
if ci_test_name == "robust_parcorr":
    from tigramite.independence_tests.robust_parcorr import RobustParCorr
    cond_ind_test = RobustParCorr()
elif ci_test_name in ("parcorr", "fisherz"):
    from tigramite.independence_tests.parcorr import ParCorr
    cond_ind_test = ParCorr()
else:
    ...  # CMIknn
```

4. **Propagar para `_pc_fallback`** analogamente.

5. **Atualizar a docstring de `discover_graph`** para documentar a nova opção.

### Resultado esperado

| Cenário | parcorr F1 | RobustParCorr F1 esperado |
|---|---|---|
| `causal_business` | 0.0 | > 0.5 (relações monotônicas detectáveis) |
| `causal_large` | 0.21 | > 0.4 |

A hipótese é que, com relações monotônicas bem definidas (adstock + Hill), o `RobustParCorr` se aproxima do oracle sem o custo computacional do CMIknn.

### Referência

Stijn Meijers, Jonas Peters, Nicolai Meinshausen (2016). "The Causal Inference via Non-Paranormal Models."
Tigramite docs: `tigramite.independence_tests.robust_parcorr`.

---

## 2. Instalar `numba` para habilitar CMIknn e GPDC

**Contexto:** `CMIknn` e `GPDC` estão implementados no tigramite mas requerem `numba`, que não está instalado no ambiente atual.

- `CMIknn`: estimador de informação mútua condicional via k-NN — teoricamente o mais adequado para dados não-lineares, mas muito lento com muitas variáveis (33 vars × max_lag=2 → ~460 testes)
- `GPDC`: Gaussian Process Distance Correlation — intermediário entre velocidade e poder

```bash
pip install numba
```

Após instalação, o `ci_test="kci"` no modo automático (1 geo) voltará a usar CMIknn nativo via tigramite, sem o fallback para causal-learn.

---

## 3. Ajuste adaptativo do `MIN_SIGMA_RATIO`

**Arquivo:** `cdnots/cdnots_model_builder.py`

O `MIN_SIGMA_RATIO=0.4` é fixo. Quando a qualidade da descoberta é baixa (muitos excluídos), comprimir todos os canais excluídos a 40% do sigma base é prejudicial — alguns deles são canais reais não detectados.

Uma heurística: ajustar `MIN_SIGMA_RATIO` dinamicamente com base na fração de canais excluídos:

```python
excluded_fraction = len(excluded_channels) / n_channels
# se muitos excluídos, punição menor (maior incerteza no grafo)
min_ratio = max(0.4, 0.4 + 0.3 * (excluded_fraction - 0.5))
```

Isso torna o prior mais conservador quando o grafo é pouco informativo.
