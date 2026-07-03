from pathlib import Path

from states import states

_DATA_DIR = Path(__file__).resolve().parent

st_123_32 = states(str(_DATA_DIR / "data/States_123_32.mat"))
st_123_41 = states(str(_DATA_DIR / "data/States_123_42.mat"))
st_123_32.plot(st_123_32.num_states - 1, 3, 2, save_path="state_123_32.svg")
st_123_41.plot(5, 3, 2, save_path="state_123_41.svg")