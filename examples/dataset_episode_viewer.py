#!/usr/bin/env python
"""Export a single LeRobot episode as an MP4 preview with camera + joint curves.

Why this script exists:
    `lerobot-dataset-viz` decodes videos with torchcodec by default and has no
    CLI flag to switch the backend. On setups where torchcodec is broken /
    incompatible with the installed PyTorch, the viewer crashes (libtorchcodec
    load errors). This script instead uses the `pyav` backend (the same backend
    that trains correctly with `--dataset.video_backend=pyav`), decodes the
    episode's camera frames, and overlays the 6 joint actions + states as
    curves below the image. The result is a plain MP4 playable in VLC / mpv /
    ffplay.

Usage:
    conda activate lerobot
    python examples/dataset_episode_viewer.py \
        --repo-id zane/pick_place_block \
        --root ./datasets/pick_place_block \
        --episode 3 \
        --out episode_3_preview.mp4

    # export all episodes 0..N
    python examples/dataset_episode_viewer.py \
        --repo-id zane/pick_place_block \
        --root ./datasets/pick_place_block \
        --episode all --out previews
"""

import argparse
from pathlib import Path

import av
import cv2
import numpy as np

from lerobot.datasets import LeRobotDataset


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Accept a frame in CHW float32 [0,1] or CHW uint8 [0,255] and return HWC uint8."""
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[1] < img.shape[2]:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    if img.dtype == np.float32 or img.dtype == np.float64:
        # If values are in [0,1] scale to [0,255]
        if img.max() <= 1.0 + 1e-6:
            img = img * 255.0
        img = img.astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def read_episode(dataset, ep_idx: int) -> dict[str, np.ndarray | list]:
    """Read all frames of one episode: images, actions, states."""
    eps = dataset.meta.episodes.to_pandas()
    row = eps[eps["episode_index"] == ep_idx].iloc[0]
    start = int(row["dataset_from_index"])
    end = int(row["dataset_to_index"]) + 1

    images = []
    actions = []
    states = []
    for idx in range(start, end):
        item = dataset[idx]
        images.append(normalize_image(np.asarray(item["observation.images.front"])))
        actions.append(np.asarray(item["action"]).flatten())
        states.append(np.asarray(item["observation.state"]).flatten())

    return {
        "images": np.stack(images),
        "actions": np.asarray(actions),
        "states": np.asarray(states),
    }


def draw_curve(canvas, values, offset_y, height, colors=None, left=10, right=None, bottom_pad=0):
    """Draw one or more normalized curves into a horizontal strip of `canvas`.

    `values` is 1D (single channel) or 2D (N, n_channels). Each channel is drawn
    with its own color from `colors` (default: a palette of 6 colors).
    """
    right = canvas.shape[1] - 10 if right is None else right
    strip_top = offset_y
    strip_bottom = offset_y + height - bottom_pad

    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    n = values.shape[0]
    n_ch = values.shape[1]
    if n < 2:
        return

    vmin = float(values.min()) if values.size else 0.0
    vmax = float(values.max()) if values.size else 1.0
    rng = (vmax - vmin) or 1.0
    span = right - left

    if colors is None:
        colors = [
            (255, 80, 80),
            (80, 255, 80),
            (80, 160, 255),
            (255, 200, 80),
            (200, 120, 255),
            (80, 255, 255),
        ]

    for ch in range(n_ch):
        color = colors[ch % len(colors)]
        pts = []
        for i in range(n):
            x = int(left + span * i / (n - 1))
            norm = (float(values[i, ch]) - vmin) / rng
            y = int(strip_bottom - norm * (strip_bottom - strip_top))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color, 1)
    return vmin, vmax


def render_episode(data: dict, fps: int = 30) -> np.ndarray:
    """Composite camera + action/state curves into a single RGB video array."""
    images = data["images"]
    actions = data["actions"]
    states = data["states"]

    n = images.shape[0]
    h, w, _ = images.shape[1:]
    panel_h = 130
    canvas_h = h + 2 * panel_h

    out = np.zeros((n, canvas_h, w, 3), dtype=np.uint8)

    for i in range(n):
        frame = np.zeros((canvas_h, w, 3), dtype=np.uint8)
        frame[:h, :, :] = images[i]

        # Action strip
        strip_top = h
        strip_bottom = h + panel_h
        cv2.rectangle(frame, (0, strip_top), (w, strip_bottom), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "ACTION (joint goals)",
            (10, strip_top + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        draw_curve(frame, actions, strip_top, panel_h, bottom_pad=20)

        # State strip
        strip_top = h + panel_h
        strip_bottom = h + 2 * panel_h
        cv2.rectangle(frame, (0, strip_top), (w, strip_bottom), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "STATE (measured joints)",
            (10, strip_top + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        draw_curve(frame, states, strip_top, panel_h, bottom_pad=20)

        # Frame counter
        cv2.putText(
            frame,
            f"frame {i}/{n}",
            (w - 90, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        out[i] = frame

    return out


def write_mp4(video: np.ndarray, out_path: str, fps: int = 30) -> None:
    """Write a numpy (N,H,W,3) uint8 video to a h264 mp4 via PyAV."""
    h, w = video.shape[1], video.shape[2]
    container = av.open(out_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "23"}

    for frame in video:
        vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
        vf = vf.reformat(width=w, height=h, format="yuv420p")
        for packet in stream.encode(vf):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Wrote {out_path} ({video.shape[0]} frames, {video.shape[1]}x{video.shape[2]})")


def main():
    parser = argparse.ArgumentParser(description="Export one or all episodes as MP4 previews.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo_id")
    parser.add_argument("--root", default=None, help="Local dataset root directory")
    parser.add_argument("--episode", required=True, help="Episode index, or 'all'")
    parser.add_argument("--out", default=None, help="Output mp4 path (or directory when --episode all)")
    parser.add_argument("--fps", type=int, default=30, help="Output video fps")
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        video_backend="pyav",  # bypass torchcodec
        return_uint8=True,
    )

    n_eps = dataset.meta.total_episodes

    if args.episode == "all":
        out_dir = Path(args.out or "episode_previews")
        out_dir.mkdir(parents=True, exist_ok=True)
        for ep in range(n_eps):
            data = read_episode(dataset, ep)
            video = render_episode(data, fps=args.fps)
            write_mp4(video, str(out_dir / f"episode_{ep:03d}.mp4"), fps=args.fps)
    else:
        ep = int(args.episode)
        out_path = args.out or f"episode_{ep:03d}_preview.mp4"
        data = read_episode(dataset, ep)
        video = render_episode(data, fps=args.fps)
        write_mp4(video, out_path, fps=args.fps)


if __name__ == "__main__":
    main()