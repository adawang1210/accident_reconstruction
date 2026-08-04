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

# Fallback anchor measurement noise, in metres, used only when a track is too
# short to estimate from. The on-contour anchor is pixel-quantised; projected to
# the ground plane that is a few centimetres.
#
# This is NOT a safe default for every track, which is why
# :func:`estimate_measurement_std` now supplies it per track. It was tuned on the
# contour-refined BMW anchor; a RAW homography anchor -- what a scene with no
# contour sidecar carries, or a vehicle whose peak-speed guard reverted to legacy
# -- is an order of magnitude noisier. Declaring 8 cm on such a track puts nearly
# every real sample outside the innovation gate, so the filter ran predict-only
# and the constant-acceleration model extrapolated freely: 7 of 13 recorded
# vehicle tracks walked off their true path, 宜蘭五結's car reversing direction
# and running 57 m the wrong way. Crucially the output stayed SMOOTH throughout
# (mean |jerk| ~4), so the jerk metrics reported it as a success.
DEFAULT_MEAS_STD_M = 0.08

# Numerical floor for the estimated noise. A zero estimate makes the measurement
# variance vanish, the Kalman gain saturate and the innovation gate divide by ~0.
MIN_MEAS_STD_M = 0.01

# How far the TYPICAL (median) smoothed sample may sit from its measurement
# before the whole track is rejected as diverged. Scaled by the track's own noise
# (a raw anchor is legitimately pulled further than a contour-refined one), with
# a floor so a very clean track still gets room to smooth. Both are deliberately
# loose: this is a net for a filter that has left the scene, not a cap on
# smoothing strength.
DIVERGENCE_TOLERANCE_M = 2.0
DIVERGENCE_TOLERANCE_SIGMA = 6.0
# ...and never more than this share of the track's own path length, which is what
# catches a short track whose noise rivals its entire motion.
DIVERGENCE_TOLERANCE_EXTENT = 0.5

# Quantile of |second difference| used to estimate a track's noise, and the
# standard-normal quantile that rescales it (Phi^-1((1+q)/2) for q = 0.9). High
# enough to see a track's noisy stretch rather than only its clean majority,
# low enough that a handful of tracking spikes do not set the level.
_NOISE_QUANTILE = 0.9
_NOISE_QUANTILE_Z = 1.6449

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


def estimate_measurement_std(positions: NDArray[np.float64]) -> float:
    """Estimate a track's per-sample measurement noise, in metres.

    Reads the noise off the track's own high-frequency content, so a raw
    homography anchor and a contour-refined one each get the ``meas_std`` they
    actually deserve instead of one constant tuned on a single clip.

    Uses the SECOND difference. At video frame rates the signal contributes
    almost nothing to it -- a 20 m-radius turn taken in 3 s implies 5.5 m/s^2,
    which over a 1/30 s step is 6 mm, far under any real anchor noise -- whereas
    white measurement noise of std sigma gives second differences of variance
    6*sigma^2.

    Takes a HIGH QUANTILE of the absolute second difference, not the median.
    Ground-plane noise is not uniform along a track: a vehicle far from the
    camera has its pixel anchor projected through a much worse-conditioned part
    of the homography, so one clip can carry centimetre noise near the camera and
    decimetre noise at the far end. A median-based estimate describes the clean
    majority and under-states the rest, which is the half of the failure the
    fixed 0.08 m constant produced -- on three recorded tracks it still left 83-91%
    of samples outside the gate. Erring high merely over-smooths a little;
    erring low lets the filter leave the data entirely.

    Args:
        positions: ``(n, 2)`` positions of one track.

    Returns:
        The estimated noise std in metres, at least :data:`MIN_MEAS_STD_M`;
        :data:`DEFAULT_MEAS_STD_M` when the track is too short (< 4 samples).

    Examples:
        ```python
        rng = np.random.default_rng(0)
        clean = np.column_stack([np.linspace(0, 30, 200), np.zeros(200)])
        round(estimate_measurement_std(clean + rng.normal(0, 0.5, clean.shape)), 1)
        # 0.5
        ```
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 4:
        return DEFAULT_MEAS_STD_M
    second = np.diff(positions, n=2, axis=0)
    # |second difference| is half-normal, so its q-quantile is sqrt(6)*sigma
    # times the standard normal's (1+q)/2 quantile -- divide that back out.
    scale = np.sqrt(6.0) * _NOISE_QUANTILE_Z
    # Per axis, then take the larger: a track can be noisier across the view than
    # along it, and under-stating the noise is the failure mode that hurts.
    estimates = [
        float(
            np.quantile(
                np.abs(second[:, axis] - np.median(second[:, axis])), _NOISE_QUANTILE
            )
        )
        / scale
        for axis in range(second.shape[1])
    ]
    return max(max(estimates), MIN_MEAS_STD_M)


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
    meas_std: float | None = None,
    process_std: float = DEFAULT_PROCESS_STD,
    gate_sigma: float = DEFAULT_GATE_SIGMA,
) -> NDArray[np.float64]:
    """Constant-acceleration Kalman + RTS smoothing of a 2D metric track.

    Args:
        frames: Frame numbers, ascending (gaps allowed -- handled via real dt).
        positions: ``(n, 2)`` metric positions.
        fps: Frames per second (used when ``times`` is None).
        times: Optional ``{frame: t_sec}`` real timestamps (PTS) for VFR clips.
        meas_std: Measurement noise std (m); larger -> smoother. Defaults to
            :func:`estimate_measurement_std` on this track -- pass a value only
            when you know the anchor's accuracy better than the data does, and
            never a constant across tracks of differing provenance.
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
    if meas_std is None:
        meas_std = estimate_measurement_std(positions)
    meas_var = meas_std * meas_std
    out = np.empty_like(positions)
    out[:, 0] = _rts_smooth_axis(positions[:, 0], dt, meas_var, process_std, gate_sigma)
    out[:, 1] = _rts_smooth_axis(positions[:, 1], dt, meas_var, process_std, gate_sigma)

    # Divergence guard, all-or-nothing per track. A smoothed path is allowed to
    # shed noise, not to leave the observation: if the filter coasted (gate
    # rejecting most samples) it produces a beautifully smooth line somewhere
    # else entirely, and no jerk metric will say so. Mirrors the peak-speed guard
    # in ``auto_reconstruct.refine_metric_from_contours`` -- when the stage cannot
    # be trusted for this vehicle, hand back what it was given.
    # MEDIAN deviation, not max: a diverged filter sits away from the data almost
    # everywhere, whereas correctly rejecting a tracking spike means leaving one
    # or two samples far behind -- which is the smoother working, not failing.
    budget = max(DIVERGENCE_TOLERANCE_M, DIVERGENCE_TOLERANCE_SIGMA * meas_std)
    # ...but never more than a fraction of how far the track actually travels.
    # Scaling the budget by sigma alone is too generous for a SHORT, NOISY track,
    # where the noise rivals the whole motion (yilan's 18-frame pedestrian: 0.39 m
    # noise over 1.8 m of walking). There the filter has no signal to lock onto,
    # trusts its prior and extrapolates -- turning a 1.4 m walk into an 11 m,
    # 71 km/h sprint. A track that has to move by a large share of its own extent
    # to look smooth is not being smoothed; it is being invented.
    path_length = float(np.hypot(*np.diff(positions, axis=0).T).sum())
    budget = min(budget, DIVERGENCE_TOLERANCE_EXTENT * path_length)
    if float(np.median(np.hypot(*(out - positions).T))) > budget:
        return positions.copy()
    return out
