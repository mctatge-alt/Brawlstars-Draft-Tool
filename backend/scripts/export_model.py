"""Export the trained PyTorch checkpoint to a NumPy archive the API can serve without torch.

    PYTHONPATH=backend python backend/scripts/export_model.py

Run this on the training machine (torch is needed here only). The API loads the resulting
``winprob.npz`` and runs inference in pure NumPy (see bsdraft/models/serve.py), so the
deployed backend needs neither torch nor the rest of the training dependencies. The npz is
tiny (~50 KB), so it's committed and ships with the repo to the cloud host.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bsdraft.constants import PROCESSED_DIR

DEFAULT_PT = PROCESSED_DIR / "winprob.pt"
DEFAULT_NPZ = PROCESSED_DIR / "winprob.npz"


def _vocab() -> dict:
    """Pin the vocabularies this export was trained against.

    Brawler rows are addressed by *position* in the reference catalog, and ranked maps are
    positional over a (mode, name)-sorted list — so refreshing ``data/reference/`` can shift an
    existing map onto a neighbour's trained row. That lookup stays in range, so no
    out-of-vocabulary guard can detect it; only the ids themselves can. Storing them here lets
    :class:`bsdraft.models.serve.WinProbModel` map id -> row exactly as trained, no matter how
    the catalog has moved since.
    """
    from bsdraft.data import encoders as E
    from bsdraft.data import reference as R
    return {
        "_vocab_brawler_ids": np.array([b.id for b in R.load_brawlers()], dtype=np.int64),
        "_vocab_map_ids": np.array([m.id for m in R.load_ranked_maps()], dtype=np.int64),
        "_vocab_map_rows": np.array([E.encode_map(m.id) for m in R.load_ranked_maps()],
                                    dtype=np.int64),
        "_vocab_modes": np.array(list(E.mode_encoder().keys())),
        "_vocab_mode_rows": np.array(list(E.mode_encoder().values()), dtype=np.int64),
    }


def export(pt_path: Path, npz_path: Path) -> None:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    weights = {k: v.detach().cpu().numpy() for k, v in ckpt["state_dict"].items()}
    np.savez(npz_path, _config=np.array(json.dumps(ckpt["config"])), **weights, **_vocab())
    size_kb = npz_path.stat().st_size / 1024
    print(f"exported {pt_path}  ->  {npz_path}  ({size_kb:.1f} KB, {len(weights)} tensors "
          f"+ pinned vocabulary)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export winprob.pt -> winprob.npz for NumPy serving.")
    ap.add_argument("--pt", type=Path, default=DEFAULT_PT, help="input torch checkpoint")
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="output NumPy archive")
    args = ap.parse_args()
    if not args.pt.exists():
        raise SystemExit(f"No checkpoint at {args.pt}. Train first: scripts/train.py")
    export(args.pt, args.npz)


if __name__ == "__main__":
    main()
