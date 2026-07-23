"""Physically-plausible trajectory smoothing: constant-acceleration Kalman + RTS.

Savitzky-Golay (:func:`ground_footprint.savgol_smooth`) is a local-polynomial
noise filter with NO physical model, so on noisy or short segments it still
wiggles and can produce kinematically implausible motion. This module adds the
standard upgrade for "physically plausible" vehicle trajectories: a
constant-acceleration (CA) Kalman filter followed by a Rauch-Tung-Striebel (RTS)
backward smoother. The CA motion model is a prior of CONTINUOUS velocity and
acceleration, so the smoothed path can't teleport or jerk implausibly, and a turn
is represented naturally (the acceleration vector points centripetally) rather
than being rounded off like a moving average would.

See ``docs/TRAJECTORY_SMOOTHING.md`` for the method survey and sources.

Everything is pure NumPy (no scipy/filterpy). x and y are smoothed independently
-- under an isotropic CA model the two axes decouple, and a turn still emerges
because each axis carries its own acceleration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Anchor measurement noise, in metres. The on-contour anchor is pixel-quantised;
# projected to the ground plane that is a few centimetres. Larger -> smoother.
DEFAULT_MEAS_STD_M = 0.08

# Process noise as a jerk "std" (m/s^3-ish); it scales the CA process covariance.
# Smaller -> the model trusts its constant-acceleration prior more -> smoother but
# laggier on genuine manoeuvres. Tuned on the BMW clip (see the module test).
DEFAULT_PROCESS_STD = 2.5

# A measurement is treated as an outlier (skipped) when it lies more than this
# many standard deviations from the model's prediction -- innovation gating.
# Lenient enough not to reject normal noise, tight enough to catch a real jump.
DEFAULT_GATE_SIGMA = 5.0

# Legacy hard cap kept for the standalone :func:`reject_kinematic_outliers`
# diagnostic; the smoother itself gates on innovation, not raw acceleration.
MAX_PLAUSIBLE_ACCEL_MPS2 = 9.0


def _dt_seconds(
    frames: list[int], fps: float, times: dict[int, float] | None
) -> NDArray[np.float64]:
    """Seconds between consecutive samples: real PTS if given, else frame/fps."""
    frame_arr = np.asarray(frames, dtype=np.float64)
    if times is not None:
        t = np.array([times[f] for f in frames], dtype=np.float64)
    else:
        t = frame_arr / max(fps, 1e-6)
    dt = np.diff(t)
    dt[dt <= 0] = 1.0 / max(fps, 1e-6)  # guard against zero/negative steps
    return dt


def trajectory_jerk(
    frames: list[int],
    positions: NDArray[np.float64],
    fps: float,
    times: dict[int, float] | None = None,
) -> dict[str, float]:
    """Kinematic-plausibility metrics of a track (higher jerk = less plausible).

    Jerk is the third time-derivative of position, estimated over consecutive
    frames. These are the standard measures used to grade reconstructed vehicle
    trajectories (see docs).

    Returns:
        ``{"mean_abs", "max_abs", "sign_changes", "frac_over_15"}`` -- mean and
        max |jerk| (m/s^3), the count of jerk sign flips (jitter shows up here),
        and the fraction of samples with |jerk| > 15 m/s^3.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(frames) < 4:
        return {
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "sign_changes": 0.0,
            "frac_over_15": 0.0,
        }
    dt = _dt_seconds(frames, fps, times)
    # Velocity, acceleration, jerk via successive differences on the (uneven) grid.
    vel = np.diff(positions, axis=0) / dt[:, None]
    dt_v = 0.5 * (dt[:-1] + dt[1:])
    acc = np.diff(vel, axis=0) / dt_v[:, None]
    dt_a = 0.5 * (dt_v[:-1] + dt_v[1:])
    jerk = np.diff(acc, axis=0) / dt_a[:, None]
    mag = np.hypot(jerk[:, 0], jerk[:, 1])
    # Sign changes of the along-path jerk component (proxy for oscillation).
    speed = np.hypot(vel[:, 0], vel[:, 1])[1:-1]
    along = np.where(
        speed > 1e-6, (jerk[:, 0] * vel[1:-1, 0] + jerk[:, 1] * vel[1:-1, 1]), 0.0
    )
    sign = np.sign(along)
    sign_changes = int(np.sum(sign[1:] * sign[:-1] < 0))
    return {
        "mean_abs": float(mag.mean()),
        "max_abs": float(mag.max()),
        "sign_changes": float(sign_changes),
        "frac_over_15": float(np.mean(mag > 15.0)),
    }


def reject_kinematic_outliers(
    frames: list[int],
    positions: NDArray[np.float64],
    fps: float,
    times: dict[int, float] | None = None,
    *,
    max_accel: float = MAX_PLAUSIBLE_ACCEL_MPS2,
    mad_k: float = 6.0,
) -> NDArray[np.bool_]:
    """Boolean mask of samples to KEEP; flags implausible-acceleration spikes.

    A point is dropped when its implied acceleration exceeds ``max_accel`` OR is a
    robust statistical outlier (median + ``mad_k``*MAD of the acceleration
    magnitude). Endpoints are always kept. Dropping these before smoothing stops a
    single tracking spike (typically across a gap) from dragging the fit.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(frames)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep
    dt = _dt_seconds(frames, fps, times)
    vel = np.diff(positions, axis=0) / dt[:, None]
    dt_v = 0.5 * (dt[:-1] + dt[1:])
    acc = np.hypot(*(np.diff(vel, axis=0) / dt_v[:, None]).T)  # (n-2,)
    median = np.median(acc)
    mad = np.median(np.abs(acc - median)) + 1e-9
    bad = (acc > max_accel) | (acc > median + mad_k * 1.4826 * mad)
    keep[1:-1] = ~bad  # acc[i] indexes interior sample i+1
    return keep


def _ca_matrices(dt: float, process_std: float):
    """Constant-acceleration transition F and process-noise Q for one step."""
    transition = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    q = process_std * process_std
    d2, d3, d4, d5 = dt**2, dt**3, dt**4, dt**5
    process = q * np.array(
        [
            [d5 / 20.0, d4 / 8.0, d3 / 6.0],
            [d4 / 8.0, d3 / 3.0, d2 / 2.0],
            [d3 / 6.0, d2 / 2.0, dt],
        ]
    )
    return transition, process


def _rts_smooth_axis(
    values: NDArray[np.float64],
    dt: NDArray[np.float64],
    meas_var: float,
    process_std: float,
    gate_sigma: float,
) -> NDArray[np.float64]:
    """CA Kalman forward filter + RTS backward smoother on one axis; return pos.

    Outliers are rejected the correct way -- by INNOVATION gating: a measurement
    is skipped (predict-only) only when it lies more than ``gate_sigma`` standard
    deviations from the model's prediction. Thresholding raw double-difference
    acceleration does NOT work: differentiating pixel noise twice makes almost
    every point look like a huge-acceleration "outlier".
    """
    n = len(values)
    dim = 3
    x = np.zeros((n, dim))
    p_cov = np.zeros((n, dim, dim))
    x_pred = np.zeros((n, dim))
    p_pred = np.zeros((n, dim, dim))
    transitions = np.zeros((n, dim, dim))
    obs = np.array([[1.0, 0.0, 0.0]])

    # Initialise position AND velocity from the head of the track. A zero-velocity
    # init makes the filter lag badly on a fast object; a single first-difference
    # is too noisy when the first step is smaller than the measurement noise (it
    # can even get the sign wrong). So fit a line through the first few points and
    # take its slope -- robust to per-frame jitter.
    cumulative_t = np.concatenate([[0.0], np.cumsum(dt)])
    head = min(6, n)
    design = np.column_stack([cumulative_t[:head], np.ones(head)])
    velocity0 = float(np.linalg.lstsq(design, values[:head], rcond=None)[0][0])
    state = np.array([values[0], velocity0, 0.0])
    cov = np.diag([meas_var, 25.0, 10.0])

    for k in range(n):
        if k == 0:
            transitions[k] = np.eye(dim)  # unused by the RTS pass (needs F[1..])
        else:
            transition, process = _ca_matrices(dt[k - 1], process_std)
            transitions[k] = transition
            state = transition @ state
            cov = transition @ cov @ transition.T + process
        x_pred[k], p_pred[k] = state, cov
        innovation = values[k] - (obs @ state)[0]
        s = (obs @ cov @ obs.T)[0, 0] + meas_var
        if innovation * innovation <= gate_sigma * gate_sigma * s:  # else: outlier
            gain = (cov @ obs.T / s)[:, 0]
            state = state + gain * innovation
            cov = cov - np.outer(gain, obs @ cov)
        x[k], p_cov[k] = state, cov

    # RTS backward pass: transitions[k+1] is the step from k to k+1.
    xs = x.copy()
    ps = p_cov.copy()
    for k in range(n - 2, -1, -1):
        c = p_cov[k] @ transitions[k + 1].T @ np.linalg.inv(p_pred[k + 1])
        xs[k] = x[k] + c @ (xs[k + 1] - x_pred[k + 1])
        ps[k] = p_cov[k] + c @ (ps[k + 1] - p_pred[k + 1]) @ c.T
    return xs[:, 0]


def kalman_rts_smooth(
    frames: list[int],
    positions: NDArray[np.float64],
    fps: float,
    times: dict[int, float] | None = None,
    *,
    meas_std: float = DEFAULT_MEAS_STD_M,
    process_std: float = DEFAULT_PROCESS_STD,
    gate_sigma: float = DEFAULT_GATE_SIGMA,
) -> NDArray[np.float64]:
    """Constant-acceleration Kalman + RTS smoothing of a 2D metric track.

    Args:
        frames: Frame numbers, ascending (gaps allowed -- handled via real dt).
        positions: ``(n, 2)`` metric positions.
        fps: Frames per second (used when ``times`` is None).
        times: Optional ``{frame: t_sec}`` real timestamps (PTS) for VFR clips.
        meas_std: Measurement noise std (m); larger -> smoother.
        process_std: Process (jerk) noise; smaller -> smoother / more prior.
        gate_sigma: Skip a measurement lying more than this many sigma from the
            prediction (innovation gating -- rejects real jumps, keeps noise).

    Returns:
        ``(n, 2)`` smoothed positions at the same frames.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(frames)
    if n < 3:
        return positions.copy()
    dt = _dt_seconds(frames, fps, times)
    meas_var = meas_std * meas_std
    out = np.empty_like(positions)
    out[:, 0] = _rts_smooth_axis(positions[:, 0], dt, meas_var, process_std, gate_sigma)
    out[:, 1] = _rts_smooth_axis(positions[:, 1], dt, meas_var, process_std, gate_sigma)
    return out
