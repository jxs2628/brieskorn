"""Thin wrapper around ./d_invariant for studying τ sequences in Python.

Usage:
    from tau import tau_sequence
    d, ks, tau = tau_sequence(2, 3, 13)
    # tau is a numpy array indexed by m = 0, …, N₀
    # where N₀ = pqr − pq − qr − pr (clamped to 0 if negative).
"""
import io
import subprocess
from pathlib import Path

import numpy as np

_BINARY = Path(__file__).resolve().parent.parent / "d_invariant"


def tau_sequence(p, q, r, binary=_BINARY):
    """Run `./d_invariant p q r --tau` and return (d, tau_array)."""
    out = subprocess.run(
        [str(binary), str(p), str(q), str(r), "--tau"],
        check=True, capture_output=True, text=True,
    ).stdout
    d = int(out.split("d         = ", 1)[1].split("\n", 1)[0])
    ks = int(out.split("k^2+s      = ", 1)[1].split("\n", 1)[0])
    tau_line = out.split("ks-2 tau\n", 1)[1].strip()
    arr = np.fromstring(tau_line, sep=" ", dtype=np.int64)
    return d, ks, arr
