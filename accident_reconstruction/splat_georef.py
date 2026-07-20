"""Splat ↔ 場景對位：用地面控制點（GCP）解相似變換。

Gaussian splat 由 SfM 定出**任意尺度／朝向／原點**的座標系；我們的場景則是「以
``origin_latlon`` 為原點的公尺」（x＝東、z＝−北、y＝上）。把兩者對齊＝一個**相似變換**
（旋轉＋等比縮放＋平移）。

給 3 個以上不共線的「splat 座標 ↔ 場景公尺座標」對應點，本模組用 **Umeyama**（1991）
閉式解求最佳相似變換，再轉成前端 ``SplatScene.tsx`` 直接吃的 ``VITE_SPLAT_*`` 環境變數
（Three.js Euler ``'XYZ'`` 旋轉、TRS 套用順序：``world = scale·R·p + pos``）。

這取代了 SPLAT_NOTES.md §5 的「盲調 .env 數字」：在 viewer 裡讀出幾個地標的 splat 座標，
配上它們在 ``reconstruction.json`` 已知的公尺座標，跑這支就得到對位數值。詳見 §9。

Example:
    >>> import numpy as np
    >>> # 一個已知的相似變換：放大 2 倍、繞 y 轉 90°、平移 (10, 0, -5)
    >>> rng = np.random.default_rng(0)
    >>> src = rng.normal(size=(5, 3))
    >>> R = threejs_euler_xyz_to_rotation_matrix(0.0, 90.0, 0.0)
    >>> dst = 2.0 * src @ R.T + np.array([10.0, 0.0, -5.0])
    >>> t = solve_splat_transform(src, dst)
    >>> round(t.scale, 6), tuple(round(a, 3) + 0.0 for a in t.rot_deg)
    (2.0, (0.0, 90.0, 0.0))
    >>> tuple(round(p, 3) + 0.0 for p in t.pos), round(t.rmse, 9)
    ((10.0, 0.0, -5.0), 0.0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Vec3 = tuple[float, float, float]

# Three.js Euler 'XYZ' 退化判斷的門檻（與 three/src/math/Euler.js 一致）。
_GIMBAL_EPS = 1.0 - 1e-7


def umeyama_similarity(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    with_scaling: bool = True,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Solve the least-squares similarity transform mapping ``source`` to ``target``.

    Implements Umeyama (1991) so that ``target ≈ scale * R @ source + t`` minimises the
    summed squared error, with ``R`` a proper rotation (no reflection, ``det(R) == 1``).

    Args:
        source: ``(n, 3)`` source points (e.g. raw splat coordinates).
        target: ``(n, 3)`` target points (e.g. our metre scene coordinates).
        with_scaling: If ``False``, fix ``scale == 1`` (rigid transform only).

    Returns:
        Tuple ``(scale, R, t)`` where ``scale`` is a float, ``R`` is ``(3, 3)`` and
        ``t`` is ``(3,)``.

    Raises:
        ValueError: If the inputs are not matching ``(n, 3)`` arrays, or fewer than
            3 correspondences are given (a 3D similarity needs ≥ 3 non-collinear pairs).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            "source and target must both be (n, 3) arrays of the same shape; "
            f"got {source.shape} and {target.shape}"
        )
    n = source.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 correspondences, got {n}")

    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean

    # Cross-covariance, then SVD; the rotation is U @ S @ Vt with a reflection guard.
    cov = (dst_centered.T @ src_centered) / n
    u, sigma, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt

    if with_scaling:
        src_var = (src_centered**2).sum() / n
        scale = float((sigma * d).sum() / src_var) if src_var > 0 else 1.0
    else:
        scale = 1.0

    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def rotation_matrix_to_threejs_euler_xyz(rotation: NDArray[np.float64]) -> Vec3:
    """Decompose a rotation matrix into Three.js Euler ``'XYZ'`` angles (degrees).

    Mirrors ``THREE.Euler.setFromRotationMatrix(m, 'XYZ')`` exactly, so feeding the
    result back as ``<primitive rotation={[x, y, z]} />`` reproduces ``rotation``
    (R3F's default Euler order is ``'XYZ'``).

    Args:
        rotation: A ``(3, 3)`` proper rotation matrix.

    Returns:
        ``(x_deg, y_deg, z_deg)`` for ``VITE_SPLAT_ROT_X/Y/Z_DEG``.
    """
    m = np.asarray(rotation, dtype=np.float64)
    y = float(np.arcsin(np.clip(m[0, 2], -1.0, 1.0)))
    if abs(m[0, 2]) < _GIMBAL_EPS:
        x = float(np.arctan2(-m[1, 2], m[2, 2]))
        z = float(np.arctan2(-m[0, 1], m[0, 0]))
    else:  # gimbal lock: |sin y| == 1, fold z into x.
        x = float(np.arctan2(m[2, 1], m[1, 1]))
        z = 0.0
    # ``+ 0.0`` returns plain Python floats and collapses -0.0 -> 0.0.
    return (
        float(np.degrees(x)) + 0.0,
        float(np.degrees(y)) + 0.0,
        float(np.degrees(z)) + 0.0,
    )


def threejs_euler_xyz_to_rotation_matrix(
    x_deg: float, y_deg: float, z_deg: float
) -> NDArray[np.float64]:
    """Build the rotation matrix for Three.js Euler ``'XYZ'`` angles (the inverse).

    Mirrors ``THREE.Matrix4.makeRotationFromEuler`` for order ``'XYZ'``. Useful for
    verifying a round-trip and for composing transforms in the same convention as the
    viewer.

    Args:
        x_deg: Rotation about x in degrees.
        y_deg: Rotation about y in degrees.
        z_deg: Rotation about z in degrees.

    Returns:
        A ``(3, 3)`` rotation matrix.
    """
    a, b = np.cos(np.radians(x_deg)), np.sin(np.radians(x_deg))
    c, d = np.cos(np.radians(y_deg)), np.sin(np.radians(y_deg))
    e, f = np.cos(np.radians(z_deg)), np.sin(np.radians(z_deg))
    ae, af, be, bf = a * e, a * f, b * e, b * f
    return np.array(
        [
            [c * e, -c * f, d],
            [af + be * d, ae - bf * d, -b * c],
            [bf - ae * d, be + af * d, a * c],
        ]
    )


@dataclass(frozen=True)
class SplatTransform:
    """A solved splat→scene similarity transform, in the viewer's own knobs.

    Attributes:
        scale: Uniform scale (``VITE_SPLAT_SCALE``); splat unit → metre.
        rot_deg: ``(x, y, z)`` Three.js Euler ``'XYZ'`` degrees (``VITE_SPLAT_ROT_*``).
        pos: ``(x, y, z)`` translation in metres (``VITE_SPLAT_X/Y/Z``).
        rmse: Root-mean-square residual of the fit, in metres (0 ⇒ exact).
    """

    scale: float
    rot_deg: Vec3
    pos: Vec3
    rmse: float

    def to_env(self) -> str:
        """Render the transform as a ``.env`` block for ``frontend/.env``."""

        def fmt(value: float) -> str:
            # Snap solver dust (sub-nm / sub-ndeg) to 0 so the .env stays readable.
            return f"{0.0 if abs(value) < 1e-9 else value:.6g}"

        rx, ry, rz = self.rot_deg
        px, py, pz = self.pos
        return "\n".join(
            (
                f"VITE_SPLAT_SCALE={self.scale:.6g}",
                f"VITE_SPLAT_ROT_X_DEG={fmt(rx)}",
                f"VITE_SPLAT_ROT_Y_DEG={fmt(ry)}",
                f"VITE_SPLAT_ROT_Z_DEG={fmt(rz)}",
                f"VITE_SPLAT_X={fmt(px)}",
                f"VITE_SPLAT_Y={fmt(py)}",
                f"VITE_SPLAT_Z={fmt(pz)}",
            )
        )


def solve_splat_transform(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> SplatTransform:
    """Solve a splat→scene transform from correspondences and report the residual.

    Args:
        source: ``(n, 3)`` splat coordinates read off the loaded splat.
        target: ``(n, 3)`` matching scene coordinates (metres; x=east, z=-north, y=up).

    Returns:
        A :class:`SplatTransform` carrying the viewer knobs and the fit ``rmse``.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scale, rotation, translation = umeyama_similarity(source, target)

    projected = scale * source @ rotation.T + translation
    rmse = float(np.sqrt(((projected - target) ** 2).sum(axis=1).mean()))

    rot_deg = rotation_matrix_to_threejs_euler_xyz(rotation)
    # ``+ 0.0`` collapses -0.0 -> 0.0 for clean .env output.
    pos = (
        float(translation[0]) + 0.0,
        float(translation[1]) + 0.0,
        float(translation[2]) + 0.0,
    )
    return SplatTransform(scale=scale, rot_deg=rot_deg, pos=pos, rmse=rmse)


def load_correspondences(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read splat↔scene correspondences from a JSON file.

    Expected shape::

        {"pairs": [{"splat": [x, y, z], "scene": [x, y, z]}, ...]}

    Args:
        path: Path to the JSON file.

    Returns:
        ``(source, target)`` arrays, each ``(n, 3)``.

    Raises:
        ValueError: If ``pairs`` is missing or any entry lacks ``splat``/``scene``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = data.get("pairs")
    if not pairs:
        raise ValueError(f"{path}: missing non-empty 'pairs' list")
    try:
        source = np.array([p["splat"] for p in pairs], dtype=np.float64)
        target = np.array([p["scene"] for p in pairs], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{path}: each pair needs 'splat' and 'scene' [x,y,z]"
        ) from exc
    return source, target


def main(correspondences: Path, env_out: Path | None = None) -> None:
    """Solve a splat→scene transform from a correspondences file and print the knobs.

    Args:
        correspondences: JSON file of splat/scene pairs (:func:`load_correspondences`).
        env_out: If given, also write the ``VITE_SPLAT_*`` block to this path.
    """
    source, target = load_correspondences(correspondences)
    transform = solve_splat_transform(source, target)
    env = transform.to_env()

    print(f"Solved similarity transform from {len(source)} correspondence(s).")
    print(f"Fit RMSE: {transform.rmse:.3f} m")
    if transform.rmse > 0.5:
        print(
            "⚠️  RMSE > 0.5 m：對應點可能讀錯、共線或不夠多；"
            "請檢查地標或多取幾個控制點。"
        )
    print("\n" + env)
    if env_out is not None:
        Path(env_out).write_text(env + "\n", encoding="utf-8")
        print(f"\nWrote -> {env_out}")


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
