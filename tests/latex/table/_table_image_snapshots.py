import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

UPDATE = os.environ.get("UPDATE_SNAPSHOTS", "").strip() in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}


def _to_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA")


def _diff_metrics(a: Image.Image, b: Image.Image) -> dict[str, float]:
    """Return mean/max absolute diff in [0,1]."""
    a = _to_rgba(a)
    b = _to_rgba(b)
    if a.size != b.size:
        return {"mean": 1.0, "max": 1.0}
    diff = ImageChops.difference(a, b)
    arr = np.asarray(diff).astype(np.float32) / 255.0
    return {"mean": float(arr.mean()), "max": float(arr.max())}


def assert_image_snapshot(
    *,
    img: Image.Image,
    name: str,
    snapshots_dir: Path,
    artifacts_dir: Path,
    tol_mean: float = 0.0025,
    tol_max: float = 0.08,
) -> None:
    """
    Compare `img` against golden `name.png` in `snapshots_dir` with tolerances.
    On failure, writes new/golden/diff into `artifacts_dir`.
    """
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    golden = snapshots_dir / f"{name}.png"

    if UPDATE or not golden.exists():
        img.save(golden)
        return

    golden_img = Image.open(golden)

    if _to_rgba(img).size != _to_rgba(golden_img).size:
        img.save(artifacts_dir / f"{name}.new.png")
        golden_img.save(artifacts_dir / f"{name}.golden.png")
        raise AssertionError(
            f"Snapshot size mismatch for {name}: new={img.size} golden={golden_img.size}. "
            f"Wrote artifacts to {artifacts_dir}"
        )

    metrics = _diff_metrics(img, golden_img)
    if metrics["mean"] > tol_mean or metrics["max"] > tol_max:
        new_path = artifacts_dir / f"{name}.new.png"
        gold_path = artifacts_dir / f"{name}.golden.png"
        diff_path = artifacts_dir / f"{name}.diff.png"

        img.save(new_path)
        golden_img.save(gold_path)
        ImageChops.difference(_to_rgba(img), _to_rgba(golden_img)).save(diff_path)

        raise AssertionError(
            f"Image snapshot mismatch for {name}: "
            f"mean={metrics['mean']:.6f} max={metrics['max']:.6f} "
            f"(tol_mean={tol_mean}, tol_max={tol_max}). "
            f"Artifacts: {artifacts_dir}"
        )
