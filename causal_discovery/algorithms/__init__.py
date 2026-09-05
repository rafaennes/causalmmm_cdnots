from causal_discovery.algorithms import (
    cdnots,
    cedar,
    dynotears,
    granger,
    lpcmci,
    pcmci_cmiknn,
    regime_pcmci,
    rpcmci,
)

REGISTRY: dict = {
    "granger": granger.discover,
    "pcmci_cmiknn": pcmci_cmiknn.discover,
    "lpcmci": lpcmci.discover,
    "dynotears": dynotears.discover,
    "cdnots": cdnots.discover,
    "regime_pcmci": regime_pcmci.discover,
    "cedar": cedar.discover,
    "rpcmci": rpcmci.discover,
}
