# Replay termination-label alignment

The world model predicts the next state from the current state and executed
action. Its termination loss must therefore compare the prediction for that
next state with the termination outcome of the same transition.

## Storage and sampling contract

`ReplayRecorder._write_rows` stores a transition in its starting-state row:

| Stored field in row t | Meaning |
| --- | --- |
| State | s_t |
| Action | a_t |
| Reward | Reward produced by the transition to s_(t+1) |
| Done | True termination produced by that same transition |

At an episode boundary, `record_pre_reset` writes the last transition and then
an extra terminal row containing its resulting state. The extra row has no
outgoing transition. It can be the last target state of a sampled window, but
cannot be a transition start. Time-limit truncations remain distinct from true
termination: the recorder uses `reset_terminated`, not the combined reset flag.

For a k-step window, states consequently come from k+1 rows, while actions,
rewards, reference frames, and termination labels all come from the first k
rows. Although a done label describes the next state, it is stored alongside
the action that produced that state, not in the next transition's row.

## Minimal counterexample

Consider two transitions: s_0 to s_1 is nonterminal, and s_1 to s_2 terminates.
The recorder writes:

| Row | State | Stored done | Terminal row? |
| --- | --- | ---: | --- |
| 0 | s_0 | 0 | No |
| 1 | s_1 | 1 | No |
| 2 | s_2 | 1 | Yes; target state only |

Sampling done from rows 1 and 2 returns `[1, 1]`, incorrectly marking the first
predicted next state as terminal. Sampling from rows 0 and 1 returns the correct
labels `[0, 1]`, aligned with the two executed actions. The duplicated label in
the extra terminal row does not justify shifting all labels by one row.

## Fix and validation

The runtime change is confined to `ReplayBuffer.sample_windows`: index done
with `ee[:-1], tt[:-1]`, just like actions and rewards. Recorder behavior,
window masks, terminal-state targets, and the world-model loss are unchanged.

`tests/test_replay_buffer.py` exercises the real recorder hooks and sampler on
CPU using a lightweight environment fixture. It covers the two-transition
regression, the last transition, truncations, terminal-row exclusion, episode
boundaries, ring wrapping, overwrite boundaries, parallel environments, and
alignment of the other transition fields. No GPU or motion dataset is needed;
the project's Python dependencies must be installed.

```sh
python -m pytest tests/test_replay_buffer.py -q
```

This is an indexing-correctness fix, not a demonstrated performance improvement.
In a matched two-seed, 320k-iteration ablation evaluated on the same 129,491
training-corpus clips, final mean normalized return changed from 0.912592 to
0.924485 for seed 0, and from 0.938566 to 0.922901 for seed 1. Failed clips changed
from 69 to 67 and from 69 to 78, respectively. These mixed results neither
establish a consistent gain nor replace the transition-alignment regression.
