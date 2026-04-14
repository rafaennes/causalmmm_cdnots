# Como Executar o Experimento de 4 Braços

---

## Opção 1 — Script Python via tmux (recomendado)

Resistente a desconexões. O experimento continua rodando mesmo que o terminal feche.

### Primeira execução

```bash
tmux new -s exp4bracos
cd /home/ennes/mestrado/causalmmm_with_cdnots
PYTHONNOUSERSITE=1 \
  /home/ennes/mestrado/pymc_meridian_comparison/.pixi/envs/default/bin/python \
    notebooks/experimento_4bracos.py 2>&1 | tee notebooks/resultados/run.log
```

Detach com `Ctrl+B` depois `D` — o processo continua rodando em background.

### Reconectar a uma sessão existente

```bash
tmux attach -t exp4bracos
```

### Ver o log sem reconectar

```bash
tail -f /home/ennes/mestrado/causalmmm_with_cdnots/notebooks/resultados/run.log
```

### Por que `PYTHONNOUSERSITE=1`?

Existe um pacote chamado `meridian` instalado em `~/.local/lib/python3.12/site-packages/` que não é o Google Meridian MMM — é uma biblioteca diferente com o mesmo nome. Sem esse flag, o Python encontra o pacote errado e falha com `ModuleNotFoundError: No module named 'meridian.model'`. O flag desativa os pacotes do usuário e força o uso do ambiente pixi.

### Checkpoints

Cada braço salva um checkpoint em `notebooks/resultados/<preset>/checkpoints/` após completar o MCMC. Se o processo for interrompido, basta reexecutar o mesmo comando — braços já concluídos são carregados do disco e não refazem o MCMC.

```
checkpoints/
  arm1_pymc_baseline.pkl
  arm2_meridian_baseline.pkl
  arm3_pymc_cdnots.pkl
  arm4_meridian_cdnots.pkl
```

Para forçar o refitting de um braço específico, basta apagar o arquivo `.pkl` correspondente.

---

## Opção 2 — Jupyter Lab

### Abrir o Jupyter Lab

```bash
cd /home/ennes/mestrado/causalmmm_with_cdnots
PYTHONNOUSERSITE=1 \
  /home/ennes/mestrado/pymc_meridian_comparison/.pixi/envs/default/bin/jupyter lab \
    --no-browser --port 8888
```

Abrir no navegador: `http://localhost:8888`

### Kernel correto

Usar o kernel **"Python 3.12 (MMM)"** — não o kernel `python3` padrão.

### Atenção

O Jupyter Lab perde o estado do kernel se a conexão cair durante o MCMC. Os checkpoints do Opção 1 também funcionam no notebook — se um braço já foi salvo como `.pkl`, a célula carrega do disco automaticamente.

---

## Regenerar o script a partir do notebook

Sempre que o notebook for modificado, regenerar o script:

```bash
PYTHONNOUSERSITE=1 \
  /home/ennes/mestrado/pymc_meridian_comparison/.pixi/envs/default/bin/jupyter nbconvert \
    --to script notebooks/experimento_4bracos.ipynb \
    --output experimento_4bracos \
    --output-dir notebooks/
```
