import matplotlib.pyplot as plt
from python_project.states import states

# Load panels once
st_103 = states("data/States_103.mat")
st_104 = states("data/States_104.mat")
st_105 = states("data/States_105.mat")
st_109 = states("data/States_109.mat")


state_idx = [0, 11, 20, 26]
freq = 3
p_idx = 2


panels = [
    ("103", st_103),
    ("104", st_104),
    ("105", st_105),
    ("109", st_109),
]
for name, state in panels:
    print(f"{name} - Total states: {state.num_states}, Frequencies: {state.num_freq}, Pair indices: {state.num_pair}")

for st_idx in state_idx:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, (name, st_obj) in zip(axes, panels):
        time = st_obj.time(st_idx, freq, p_idx)
        amp = st_obj.amplitude(st_idx, freq, p_idx)
        ax.plot(time, amp, label=f"State {st_idx} Panel {name}")
        ax.set_title(f"Panel {name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend()

    fig.suptitle(f"Signals for state {st_idx} (freq {freq}, pair {p_idx})")
    fig.tight_layout()

plt.clf()

st_123_41 = states("data/States_123_41.mat")
st_123_41.plot(st_123_41.num_states - 1, 3, 2)
plt.show()