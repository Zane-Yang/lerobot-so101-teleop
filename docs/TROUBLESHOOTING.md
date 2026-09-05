# Troubleshooting Notes

This file records **real problems met during development** and the concrete
fixes used. It is meant to give a reviewer concrete evidence of engineering
judgment (environment debugging, data consistency, hardware reliability), not
just a “copy CLI and it works” README.

---

## 1. `torchcodec` could not be loaded (PyTorch 2.7 + torchcodec mismatch)

### Symptom

```text
RuntimeError: Could not load libtorchcodec ...
OSError: /home/.../torchcodec/libtorchcodec_core7.so:
        undefined symbol: torch_dtype_float4_e2m1fn_x2
```

Every code path that decodes videos (training data loader,
`lerobot-dataset-viz`, edit tools) would fail on the first batch.

### Root cause

The installed `torchcodec` wheel is not binary-compatible with the installed
PyTorch (2.7.1+cu128). The environment predates the project; we did not want to
mutate the PyTorch install under an in-use conda env.

### Fix

1. Pin all data consumers to the **pyAV** decode backend, which the framework
   supports:
   - Training: `--dataset.video_backend=pyav`
   - Custom episode preview: `examples/dataset_episode_viewer.py` constructs
     `LeRobotDataset(..., video_backend="pyav")`.
2. For tools with **no backend switch** (e.g. `lerobot-dataset-viz`), we ship a
   small pyAV-based alternative instead of “fixing” a broken C++ extension.

**Lesson**: pin video-backend dependency explicitly in README to make the setup
deterministic across machines.

---

## 2. Interrupted session left episode metadata with wrong `to_timestamp`

### Symptom

`lerobot-edit-dataset delete_episodes` crashed with:

```text
AssertionError: Episode length mismatch: 603 vs 884
```

### Root cause

Recording was interrupted (camera USB dropout) then resumed. For 4 episodes the
`to_timestamp` in `meta/episodes/*.parquet` no longer matched
`length / fps` — it had swallowed some consecutive “episode boundary” seconds.
The frames themselves were fine; only the index was wrong.

### Diagnosis (showing a systematic check)

```python
for each episode:
    expected = round(to_timestamp * fps) - round(from_timestamp * fps)
    assert length == expected
```

We found exactly 4 mismatches: ep 11, 27, 30, 47.

### Fix

If `data` parquet rows agree with `length` (i.e., length is the source of
truth), rewrite `to_timestamp = from_timestamp + length / fps` in the episode
parquet. After write-back, re-ran the consistency check: 0 mismatches, then
`delete_episodes` succeeded (55 → 48 episodes, 25942 frames).

**Lesson**: when an “index assertion” fails before doing anything destructive,
treat it as **metadata out of sync with content**, verify which side is the
source of truth, then repair the metadata — never silently skip the check.

---

## 3. USB camera dropout during long recording

### Symptom

```text
OpenCVCamera(2) read failed (status=False)
VIDIOC_REQBUFS: errno=19 (No such device)
```

### Root cause

The USB webcam disappeared from the kernel (log shows the device node removed).
Most likely: cable/port contact, power budget on a laptop USB hub, or kernel
auto-suspend.

### Mitigation

- Hardware first: use a short quality USB cable directly into the laptop, test
  5 min before starting an N×30 s recording session.
- Disable USB autosuspend (one-off test):
  ```bash
  echo -1 | sudo tee /sys/bus/usb/devices/*/power/autosuspend_delay_ms
  ```
- Resume from state: `lerobot-record --resume=true` continues writing so the
  already-recorded episodes are not lost.

---

## 4. Gamepad works physically but `gamepad_joints` not listed in CLI

### Symptom

```text
lerobot-record: error: argument --teleop.type: invalid choice: 'gamepad_joints'
```

even though `lerobot-teleoperate` accepted it.

### Root cause

The CLI choices are generated from **registered class subclasses discovered by
import**; `lerobot-record`'s module imported the teleop namespace in a way that
did not import `lerobot.teleoperators.gamepad`, so the subclass was never
registered (CPython registration happens at import-time).

### Fix

Ensure the module that runs `lerobot-record` triggers importing
`teleoperators.gamepad` (same objects used by teleoperate). After the import,
the factory and argparse both recognize `gamepad_joints`.

**Lesson**: when adding a teleoperator, make sure every entry script imports the
module that runs the `@register_subclass` decorator.