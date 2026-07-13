import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "algorithms", "training", "viz", "scripts"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from states import states
from config import mat_file_path, DEFAULT_FREQ_INDEX

st_123_32 = states(str(mat_file_path("123_32")))
st_123_41 = states(str(mat_file_path("123_42")))
st_123_32.plot(st_123_32.num_states - 1, DEFAULT_FREQ_INDEX, 2, save_path="state_123_32.svg")
st_123_41.plot(5, DEFAULT_FREQ_INDEX, 2, save_path="state_123_41.svg")