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

# --- Shape smoothing (:func:`fit_smooth_curve`) -------------------------------
#
# The Kalman+RTS pass above is judged on POSITION error and on jerk, and it wins
# on both -- yet the drawn route still comes out visibly saw-toothed. Those two
# facts are not in conflict. Jerk is a TIME derivative, so it is divided by dt^3;
# on a vehicle that barely moves (the BMW car covers 5.5 m in 192 frames -- 2.9 cm
# per frame, at a residual noise of ~2 cm) a few-centimetre wobble is a tiny jerk
# and a tiny position error, but it is a LARGE change of heading, because the
# heading is the wobble divided by the step, and the step is the same size as the
# wobble. That heading flip, frame after frame, is exactly the zig-zag the reader
# sees. So "smooth" has to be enforced on the SHAPE (turn angle per sample), not
# only on the kinematics.
#
# p99 of the per-sample turn angle we aim to get under. A real manoeuvre is far
# gentler than this: a car turning 90 degrees in one second at 25 fps sweeps 3.6
# degrees per sample, so a 5-degree target rounds off jitter without flattening
# any turn a vehicle can physically make.
CURVE_TURN_TARGET_DEG = 5.0

# Turn angle per SAMPLE is blind to the other way a route looks unsmooth: an
# elbow. Heading that swings 20 degrees over eight samples turns only 2.5 degrees
# each, passes the test above, and still draws as a corner joining two straight
# runs. What rules an elbow out is not geometry but physics -- taking it at the
# speed the track claims would need a lateral acceleration no road vehicle on dry
# asphalt can produce. 4 m/s^2 (~0.4 g) is the standard comfortable-to-firm
# cornering limit; anything above it in a reconstructed route is the calibration
# or the tracker, not the driver.
#
# This criterion is also self-guarding where the turn-angle one is not: lateral
# acceleration is (yaw rate x speed), so a stopped vehicle's meaningless heading
# flips score ~0 instead of vetoing every candidate fit.
MAX_LATERAL_ACCEL_MPS2 = 4.0

# Log-spaced roughness penalties tried, weakest first. Wide on purpose: lambda
# carries units (position^2 / (position/time^2)^2 * time), so its useful value
# shifts with fps and track extent; :func:`fit_smooth_curve` picks from the grid
# by measured behaviour rather than us pre-tuning a number per scene.
_CURVE_LAMBDAS = np.logspace(-8.0, 8.0, 65)

# A segment shorter than this fraction of the track's median step has no
# meaningful direction, so its turn angle is noise about noise; excluded from the
# percentile rather than allowed to veto every candidate.
_MIN_SEGMENT_FRACTION = 0.05


def turn_angles_deg(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Heading change (degrees, 0-180) between each pair of consecutive segments.

    The direct measure of how saw-toothed a drawn route looks: a smooth curve
    turns by a small amount per sample, a zig-zag alternates by tens of degrees
    however small its position error is.

    Args:
        positions: ``(n, 2)`` positions of one track.

    Returns:
        ``(m,)`` angles for the segments long enough to have a direction
        (see :data:`_MIN_SEGMENT_FRACTION`); empty when the track is too short
        or degenerate.

    Examples:
        ```python
        straight = np.column_stack([np.arange(5.0), np.zeros(5)])
        turn_angles_deg(straight).round(6).tolist()
        # [0.0, 0.0, 0.0]
        ```
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 3:
        return np.empty(0)
    segments = np.diff(positions, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    median_step = float(np.median(lengths))
    floor = max(_MIN_SEGMENT_FRACTION * median_step, 1e-12)
    headings = np.degrees(np.arctan2(segments[:, 1], segments[:, 0]))
    change = np.abs(np.diff(headings))
    change = np.minimum(change, 360.0 - change)
    usable = (lengths[:-1] > floor) & (lengths[1:] > floor)
    return change[usable]


def lateral_accelerations(
    frames: list[int],
    positions: NDArray[np.float64],
    fps: float,
    times: dict[int, float] | None = None,
) -> NDArray[np.float64]:
    """Lateral (centripetal) acceleration at each interior sample, in m/s^2.

    ``a_lat = yaw_rate * speed`` -- the cornering effort the reconstructed route
    attributes to the vehicle. Used to reject an elbow: a corner that would need
    more than :data:`MAX_LATERAL_ACCEL_MPS2` did not happen, whatever its
    per-sample turn angle looks like.

    Args:
        frames: Frame numbers, ascending (gaps allowed -- handled via real dt).
        positions: ``(n, 2)`` positions of one track.
        fps: Frames per second (used when ``times`` is None).
        times: Optional ``{frame: t_sec}`` real timestamps (PTS) for VFR clips.

    Returns:
        ``(n-2,)`` lateral accelerations; empty when the track is too short.

    Examples:
        ```python
        # A straight run corners not at all.
        straight = np.column_stack([np.arange(5.0), np.zeros(5)])
        lateral_accelerations(list(range(5)), straight, 10.0).round(6).tolist()
        # [0.0, 0.0, 0.0]
        ```
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 3 or len(frames) != len(positions):
        return np.empty(0)
    dt = _dt_seconds(frames, fps, times)
    segments = np.diff(positions, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    speeds = lengths / dt
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    # Yaw rate over the midpoint spacing, and the speed there -- both centred on
    # the same interior sample so their product is that sample's cornering.
    dt_mid = 0.5 * (dt[:-1] + dt[1:])
    yaw_rate = np.abs(np.diff(headings)) / dt_mid
    mid_speed = 0.5 * (speeds[:-1] + speeds[1:])
    return yaw_rate * mid_speed


def _second_derivative_operator(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(n-2, n)`` finite-difference second derivative on an uneven time grid.

    Built from the real timestamps rather than sample index, so a bridged
    tracking gap is penalised for the time it spans instead of counting as one
    ordinary step.
    """
    n = len(t)
    operator = np.zeros((n - 2, n))
    h1 = t[1:-1] - t[:-2]
    h2 = t[2:] - t[1:-1]
    rows = np.arange(n - 2)
    operator[rows, rows] = 2.0 / (h1 * (h1 + h2))
    operator[rows, rows + 1] = -2.0 / (h1 * h2)
    operator[rows, rows + 2] = 2.0 / (h2 * (h1 + h2))
    return operator


def fit_smooth_curve(
    frames: list[int],
    positions: NDArray[np.float64],
    fps: float,
    times: dict[int, float] | None = None,
    *,
    measurements: NDArray[np.float64] | None = None,
    turn_target_deg: float = CURVE_TURN_TARGET_DEG,
    meas_std: float | None = None,
) -> NDArray[np.float64]:
    """Fit the smoothest curve through a track that the measurements still allow.

    A penalised least-squares (Whittaker) fit: minimise
    ``||z - y||^2 + lambda * ||z''||^2`` on the real time grid, jointly for x and
    y. It is the discrete smoothing spline, so the solution is a cubic-spline-like
    C^2 curve; as ``lambda`` grows the second derivative is driven to zero and the
    fit degenerates to a STRAIGHT LINE, which is why a straight run comes out
    straight and a turn comes out as an actual arc.

    ``lambda`` is not tuned per scene. Two opposing quantities bracket it -- the
    p95 turn angle FALLS as lambda grows, the deviation from the measurements
    RISES -- so the grid is walked from the weakest penalty upward, and the fit
    returned is the first one whose shape meets ``turn_target_deg``, or, if none
    does, the strongest one still inside the faithfulness budget. The budget is
    the same all-or-nothing rule :func:`kalman_rts_smooth` uses (median deviation
    against :data:`DIVERGENCE_TOLERANCE_M` / ``SIGMA`` / ``EXTENT``), so this
    stage can round off a zig-zag but cannot walk the route off the road.

    Args:
        frames: Frame numbers, ascending (gaps allowed -- handled via real dt).
        positions: ``(n, 2)`` positions to smooth (the Kalman+RTS output).
        fps: Frames per second (used when ``times`` is None).
        times: Optional ``{frame: t_sec}`` real timestamps (PTS) for VFR clips.
        measurements: ``(n, 2)`` positions the fit must stay faithful to --
            the anchors that went INTO the smoother. Defaults to ``positions``.
        turn_target_deg: p95 per-sample turn angle to get under.
        meas_std: Measurement noise (m) sizing the budget; estimated from
            ``measurements`` when None.

    Returns:
        ``(n, 2)`` smoothed positions at the same frames.

    Examples:
        ```python
        rng = np.random.default_rng(0)
        frames = list(range(60))
        line = np.column_stack([np.linspace(0, 6, 60), np.zeros(60)])
        noisy = line + rng.normal(0, 0.05, line.shape)
        fit = fit_smooth_curve(frames, noisy, 30.0)
        bool(np.percentile(turn_angles_deg(fit), 95) < 5.0)
        # True
        ```
    """
    positions = np.asarray(positions, dtype=np.float64)
    reference = (
        positions
        if measurements is None
        else np.asarray(measurements, dtype=np.float64)
    )
    n = len(frames)
    if n < 4 or len(reference) != n:
        return positions.copy()

    dt = _dt_seconds(frames, fps, times)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    operator = _second_derivative_operator(t)
    # Trapezoidal weights: penalise the integral of the squared second
    # derivative, so an unevenly sampled stretch is not penalised more heavily
    # simply for carrying more samples per second.
    weights = 0.5 * (t[2:] - t[:-2])
    penalty = operator.T @ (weights[:, None] * operator)
    identity = np.eye(n)

    if meas_std is None:
        meas_std = estimate_measurement_std(reference)
    budget = max(DIVERGENCE_TOLERANCE_M, DIVERGENCE_TOLERANCE_SIGMA * meas_std)
    path_length = float(np.hypot(*np.diff(reference, axis=0).T).sum())
    budget = min(budget, DIVERGENCE_TOLERANCE_EXTENT * path_length)

    best = positions.copy()
    for lam in _CURVE_LAMBDAS:
        try:
            candidate = np.linalg.solve(identity + lam * penalty, positions)
        except np.linalg.LinAlgError:  # pragma: no cover - singular is not reachable
            break
        if float(np.median(np.hypot(*(candidate - reference).T))) > budget:
            break  # past here every stronger penalty is further out too
        best = candidate
        # Both defects have to be gone: per-sample zig-zag AND the elbow. Judged
        # at p99, not p95 or a mean, because either one is a LOCAL defect -- a
        # 285-sample track whose p95 passes still leaves ~14 samples free to
        # kink, and a single corner in an otherwise clean line is exactly what
        # the reader's eye lands on.
        angles = turn_angles_deg(candidate)
        cornering = lateral_accelerations(frames, candidate, fps, times)
        zigzag = angles.size and float(np.percentile(angles, 99)) > turn_target_deg
        # MAX for the cornering, not a percentile: the sharpest corner a route can
        # have is a single sample wide, and on a 120-sample track a p99 steps
        # straight over it. Turn angle keeps its percentile because its
        # short-segment rule is a heuristic that a max would let dominate;
        # lateral acceleration needs no such rule (a stopped vehicle scores ~0),
        # so its worst sample is trustworthy on its own.
        elbow = cornering.size and float(cornering.max()) > MAX_LATERAL_ACCEL_MPS2
        if not zigzag and not elbow:
            return candidate
    return best


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
