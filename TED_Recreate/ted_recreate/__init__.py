"""Re-creation of how TED (The Encyclopedia of Domains) chops proteins into domains and gives them CATH labels."""
import ctypes
import sys
from pathlib import Path

# In a conda environment, pyarrow (pulled in by transformers when ESMFold is imported) needs a newer libstdc++ than
# the system one, and whichever copy is loaded first wins. Load the environment's copy before anything else does.
try:
    ctypes.CDLL(str(Path(sys.executable).resolve().parents[1] / "lib/libstdc++.so.6"), mode=ctypes.RTLD_GLOBAL)
except OSError:     # not a conda environment: nothing to do
    pass

from .chop import ChopResult, Domain, ted_chop, ted_chop_batch  # noqa: E402
from .classify import Classifier, LabelResult, cath_label, cath_label_chopped  # noqa: E402

__all__ = ["ted_chop", "ted_chop_batch", "cath_label", "cath_label_chopped", "Classifier", "ChopResult", "Domain",
           "LabelResult"]
