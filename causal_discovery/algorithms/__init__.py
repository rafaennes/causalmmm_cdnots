from causal_discovery.algorithms import cdnots, dynotears, granger, lpcmci, pcmci_cmiknn

REGISTRY: dict = {
    "granger": granger.discover,
    "pcmci_cmiknn": pcmci_cmiknn.discover,
    "lpcmci": lpcmci.discover,
    "dynotears": dynotears.discover,
    "cdnots": cdnots.discover,
}
