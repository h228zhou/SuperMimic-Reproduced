"""Recorder-to-sampler termination alignment using deterministic CPU transitions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.tracking_env_cfg import ReplayRecorderCfg


def make_recorder(capacity=16, num_envs=1):
    environment = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        state=torch.zeros(num_envs, 3),
        context=torch.arange(num_envs * 2, dtype=torch.float32).reshape(num_envs, 2),
        reward_buf=torch.zeros(num_envs),
        reset_buf=torch.zeros(num_envs, dtype=torch.bool),
        reset_terminated=torch.zeros(num_envs, dtype=torch.bool),
    )
    command = SimpleNamespace(
        frame=torch.zeros(num_envs, dtype=torch.long),
        library=SimpleNamespace(frame_clip=torch.zeros(4096, dtype=torch.long)),
    )
    action = SimpleNamespace(executed_raw=torch.zeros(num_envs, 29))
    environment.command_manager = SimpleNamespace(get_term=lambda name: command)
    environment.action_manager = SimpleNamespace(
        action=torch.zeros(num_envs, 29), get_term=lambda name: action
    )
    recorder = ReplayRecorderCfg.func.__new__(ReplayRecorderCfg.func)
    recorder._env = environment
    recorder.state_io = SimpleNamespace(read=lambda: environment.state.clone())
    recorder.ctx = SimpleNamespace(read=lambda: environment.context.clone())
    recorder.command_name = "motion"
    recorder.action_name = "joint_pos"
    recorder.buffer = ReplayBuffer(num_envs, capacity, 3, 29, 2, "cpu")
    recorder.prev_s = torch.zeros(num_envs, 3)
    recorder.prev_xi = torch.zeros(num_envs, 2)
    recorder.prev_frame = torch.zeros(num_envs, dtype=torch.long)
    recorder.prev_clip = torch.zeros(num_envs, dtype=torch.long)
    recorder.ep_start = torch.ones(num_envs, dtype=torch.bool)
    recorder.enabled = True
    recorder.record_post_reset(torch.arange(num_envs))
    return recorder


def advance(recorder, terminated=(), truncated=()):
    environment = recorder._env
    environment.action_manager.action[:] = environment.state[:, :1] + 1
    recorder.action_term.executed_raw[:] = environment.action_manager.action + 100
    environment.reward_buf[:] = (environment.state[:, 0] + 1) * 10
    environment.state += 1
    recorder.command.frame += 1
    environment.reset_buf.zero_()
    environment.reset_terminated.zero_()
    for index in terminated:
        environment.reset_buf[index] = True
        environment.reset_terminated[index] = True
    for index in truncated:
        environment.reset_buf[index] = True
    reset_ids = environment.reset_buf.nonzero().flatten()
    recorder.record_pre_reset(reset_ids)
    recorder.record_post_step()


def select_window(buffer, start, horizon, env_index=0):
    assert buffer.window_mask(horizon)[env_index, start].item()
    with patch.object(
        buffer,
        "_sample_from_mask",
        return_value=(torch.tensor([env_index]), torch.tensor([start])),
    ):
        return buffer.sample_windows(1, horizon)


def assert_labels(window, expected):
    torch.testing.assert_close(window.done[:, 0], torch.tensor(expected, dtype=torch.float32))


def test_exact_two_transition_termination_regression():
    recorder = make_recorder()
    advance(recorder)
    advance(recorder, terminated=(0,))
    window = select_window(recorder.buffer, 0, 2)
    assert_labels(window, [0, 1])
    torch.testing.assert_close(window.states[:, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(window.a_clean[:, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(window.a_exec[:, 0, 0], torch.tensor([101.0, 102.0]))
    torch.testing.assert_close(window.reward[:, 0], torch.tensor([10.0, 20.0]))
    torch.testing.assert_close(window.frame[:, 0], torch.tensor([0, 1]))
    torch.testing.assert_close(window.xi, recorder._env.context)


def test_penultimate_transition_is_not_terminal():
    recorder = make_recorder()
    advance(recorder)
    advance(recorder, terminated=(0,))
    assert_labels(select_window(recorder.buffer, 0, 1), [0])


def test_last_transition_is_terminal():
    recorder = make_recorder()
    advance(recorder, terminated=(0,))
    assert_labels(select_window(recorder.buffer, 0, 1), [1])


def test_nonterminal_window():
    recorder = make_recorder()
    for _ in range(3):
        advance(recorder)
    assert_labels(select_window(recorder.buffer, 0, 2), [0, 0])


def test_timeout_is_not_true_termination():
    recorder = make_recorder()
    advance(recorder)
    advance(recorder, truncated=(0,))
    assert_labels(select_window(recorder.buffer, 0, 2), [0, 0])
    assert recorder.buffer.terminal[0, 2].item()


def test_simultaneous_timeout_and_termination_retains_true_termination():
    recorder = make_recorder()
    advance(recorder)
    advance(recorder, terminated=(0,), truncated=(0,))
    assert_labels(select_window(recorder.buffer, 0, 2), [0, 1])


def test_terminal_state_is_a_target_not_a_transition_start():
    recorder = make_recorder()
    advance(recorder, terminated=(0,))
    assert recorder.buffer.terminal[0, 1].item()
    assert not recorder.buffer.window_mask(1)[0, 1].item()
    assert_labels(select_window(recorder.buffer, 0, 1), [1])
    assert recorder.buffer.total_rows == 2


@pytest.mark.parametrize("terminated", [False, True])
def test_windows_do_not_cross_reset_boundaries(terminated):
    recorder = make_recorder()
    advance(recorder)
    advance(recorder, terminated=(0,) if terminated else (), truncated=() if terminated else (0,))
    recorder._env.state.fill_(100)
    recorder.command.frame.fill_(100)
    recorder.record_post_reset(torch.tensor([0]))
    for _ in range(3):
        advance(recorder)
    valid = recorder.buffer.window_mask(2)[0]
    assert valid.nonzero().flatten().tolist() == [0, 3]
    assert_labels(select_window(recorder.buffer, 0, 2), [0, int(terminated)])
    assert_labels(select_window(recorder.buffer, 3, 2), [0, 0])
    assert recorder.buffer.ep_id[0, 2] != recorder.buffer.ep_id[0, 3]
    assert recorder.buffer.ep_start[0, 3].item()


@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_ring_wrap_preserves_transition_alignment(terminated, horizon):
    recorder = make_recorder(capacity=5)
    for _ in range(7):
        advance(recorder)
    advance(recorder, terminated=(0,) if terminated else (), truncated=() if terminated else (0,))
    buffer = recorder.buffer
    valid_starts = buffer.window_mask(horizon)[0].nonzero().flatten().tolist()
    assert valid_starts
    assert any(start + horizon >= buffer.T for start in valid_starts)
    for start in valid_starts:
        window = select_window(buffer, start, horizon)
        torch.testing.assert_close(
            window.states[1:, 0, 0] - window.states[:-1, 0, 0], torch.ones(horizon)
        )
        expected = ((window.states[1:, 0, 0] == 8) & terminated).float()
        torch.testing.assert_close(window.done[:, 0], expected)
        torch.testing.assert_close(window.a_clean[:, 0, 0], window.states[:-1, 0, 0] + 1)
        torch.testing.assert_close(window.reward[:, 0], (window.states[:-1, 0, 0] + 1) * 10)
        torch.testing.assert_close(window.frame[:, 0], window.states[:-1, 0, 0].long())


def test_same_episode_write_pointer_cannot_join_newest_to_oldest():
    recorder = make_recorder(capacity=5)
    for _ in range(12):
        advance(recorder)
    buffer = recorder.buffer
    for horizon in (1, 2, 3):
        for start in buffer.window_mask(horizon)[0].nonzero().flatten().tolist():
            window = select_window(buffer, start, horizon)
            torch.testing.assert_close(
                window.states[1:] - window.states[:-1], torch.ones_like(window.states[1:])
            )
            assert_labels(window, [0] * horizon)


def test_parallel_environments_keep_distinct_terminal_labels():
    recorder = make_recorder(num_envs=3)
    advance(recorder)
    advance(recorder, terminated=(0,), truncated=(1,))
    recorder._env.state[:2].fill_(100)
    recorder.command.frame[:2].fill_(100)
    recorder.record_post_reset(torch.tensor([0, 1]))
    advance(recorder)
    assert_labels(select_window(recorder.buffer, 0, 2, 0), [0, 1])
    assert_labels(select_window(recorder.buffer, 0, 2, 1), [0, 0])
    assert_labels(select_window(recorder.buffer, 0, 2, 2), [0, 0])


def test_empty_buffer_has_no_sampleable_window():
    with pytest.raises(RuntimeError, match="no sampleable rows"):
        ReplayBuffer(1, 5, 3, 29, 2, "cpu").sample_windows(1, 2)
