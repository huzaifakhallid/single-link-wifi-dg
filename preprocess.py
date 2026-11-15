"""
Signal pipeline for single-link Widar3.0 CSI (receiver r1, antennas 1 and 2, one TX stream).

  raw .dat --csiread--> (packets, 30, 3, 1) complex
    -> sort / de-duplicate timestamps (uint32 microsecond counter, wrap = backward step > 2^31)
    -> Hampel filter (placement is a pipeline option)
    -> CSI ratio R = H1 / (H2 + eps)
    -> uniform 1 kHz grid (linear interpolation) -> polyphase decimation to 200 Hz
    -> phase map (30 x T) and Doppler map (F x T'), each resized to 64 x 128, float16

Every stage that has more than one candidate implementation is selected by name, so the
diagnostic study (diagnose.py) and the locked pipeline (LOCKED below) run the same code.
No stage has a fallback: a trial either yields features or an explicit exclusion reason.
"""
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal

NUM_SUBCARRIERS = 30
RAW_GRID_HZ = 1000.0      # nominal Intel 5300 packet rate in Widar3.0
FS = 200.0                # analysis sample rate
MAP_SHAPE = (64, 128)     # (frequency / subcarrier axis, time axis)
MAX_GAP_S = 0.100
MIN_DUR_S = 0.500
STFT_WIN, STFT_HOP, STFT_NFFT, DOPPLER_MAX_HZ = 64, 16, 256, 60.0
BAND_HZ = (2.0, 60.0)
DB_FLOOR = -40.0

HAMPEL_MODES = ("antenna_complex", "ratio")
PHASE_VARIANTS = ("unwrap", "unwrap_hp", "rel", "diff")
DOPPLER_VARIANTS = ("mean_avg_log1p", "mean_avg_db", "bp_avg_log1p", "bp_avg_db", "bp_pca_log1p", "bp_pca_db")

# Locked pipeline. Set once from the diagnostic gate (results_diag/gate_table.csv, docs/decisions.md)
# and never changed afterwards; every cached feature file records the values it was built with.
LOCKED: Dict[str, str] = {
    "hampel_mode": "ratio",
    "phase_variant": "diff",
    "doppler_variant": "bp_pca_log1p",
}


# --------------------------------------------------------------------------- decoding
def load_trial(path: str):
    """Decode one Intel-5300 trial. Returns (csi (N,30,3,1) complex, timestamp_low uint32, info)."""
    import csiread
    c = csiread.Intel(str(path), nrxnum=3, ntxnum=1, if_report=False)
    c.read()
    n = int(c.count)
    if n == 0:
        return None, None, {"n_packets_raw": 0, "n_dropped_nrx": 0}
    keep = np.asarray(c.Nrx[:n]) == 3  # antennas 1 and 2 are only defined when all 3 RF chains reported
    csi = c.get_scaled_csi()[:n][keep]
    ts = np.asarray(c.timestamp_low[:n], dtype=np.uint32)[keep]
    assert csi.ndim == 4 and csi.shape[1:] == (30, 3, 1), f"Unexpected CSI shape {csi.shape} in {path}"
    return csi, ts, {"n_packets_raw": n, "n_dropped_nrx": int((~keep).sum())}


def unwrap_timestamps(ts_low: np.ndarray) -> Tuple[np.ndarray, int]:
    """uint32 microsecond counter -> int64. A wrap is a backward step larger than 2^31 us; smaller
    backward steps are packet reordering and are left for the sort."""
    ts = np.asarray(ts_low).astype(np.int64)
    assert ts.min() >= 0 and ts.max() < (1 << 32), "timestamp_low outside uint32 range"
    wraps = (np.diff(ts) < -(1 << 31)).astype(np.int64)
    return ts + np.concatenate(([0], np.cumsum(wraps))) * (1 << 32), int(wraps.sum())


# --------------------------------------------------------------------------- cleaning
def hampel_filter(x: np.ndarray, window_size: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Vectorised Hampel along axis 0 of a real (T, S) array: replace a sample by the window median
    when |x - med| > n_sigmas * 1.4826 * MAD. Reflect padding at the edges."""
    k = window_size // 2
    xp = np.pad(x, ((k, k), (0, 0)), mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(xp, window_size, axis=0)
    med = np.median(win, axis=-1)
    mad = np.median(np.abs(win - med[..., None]), axis=-1)
    return np.where(np.abs(x - med) > n_sigmas * 1.4826 * mad, med, x)


def hampel_filter_complex(z: np.ndarray, window_size: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    return hampel_filter(z.real, window_size, n_sigmas) + 1j * hampel_filter(z.imag, window_size, n_sigmas)


def compute_csi_ratio(ant1: np.ndarray, ant2: np.ndarray) -> np.ndarray:
    """R = H1 / (H2 + eps), eps = 1e-6 * median|H2| (only guards an exact zero denominator)."""
    eps = 1e-6 * (np.median(np.abs(ant2)) + 1e-12)
    return ant1 / (ant2 + eps)


def resample_uniform(t_seconds: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Non-uniform packets -> 1 kHz grid (linear interpolation) -> resample_poly /5 -> 200 Hz.
    resample_poly applies its Kaiser FIR low-pass at the 1 kHz rate, i.e. anti-aliasing."""
    n_hi = int(np.floor(t_seconds[-1] * RAW_GRID_HZ)) + 1
    t_hi = np.arange(n_hi) / RAW_GRID_HZ
    hi = np.empty((n_hi, data.shape[1]), dtype=np.complex128)
    for s in range(data.shape[1]):
        hi[:, s] = np.interp(t_hi, t_seconds, data[:, s].real) + 1j * np.interp(t_hi, t_seconds, data[:, s].imag)
    down = int(round(RAW_GRID_HZ / FS))
    assert abs(RAW_GRID_HZ / down - FS) < 1e-9
    return scipy.signal.resample_poly(hi, up=1, down=down, axis=0)


def ratio_200hz(csi: Optional[np.ndarray], timestamps: Optional[np.ndarray], hampel_mode: str):
    """Raw packets -> CSI ratio on a uniform 200 Hz grid. Returns (R (T,30) complex128 or None, meta);
    meta['exclusion_reason'] is None for a usable trial."""
    assert hampel_mode in HAMPEL_MODES, hampel_mode
    meta: Dict[str, Any] = {"exclusion_reason": None, "hampel_mode": hampel_mode}
    if csi is None or len(csi) == 0:
        meta["exclusion_reason"] = "empty: 0 packets"
        return None, meta
    assert csi.ndim == 4 and csi.shape[1:3] == (30, 3), f"Unexpected CSI shape {csi.shape}"
    meta["n_packets"] = int(csi.shape[0])
    if csi.shape[0] < 50:
        meta["exclusion_reason"] = f"too_few_packets: {csi.shape[0]}"
        return None, meta

    ts, n_wraps = unwrap_timestamps(timestamps)
    order = np.argsort(ts, kind="stable")
    ts, csi = ts[order], csi[order]
    uniq = np.concatenate(([True], np.diff(ts) > 0))
    ts, csi = ts[uniq], csi[uniq]
    t_sec = (ts - ts[0]) / 1e6
    dur = float(t_sec[-1])
    max_gap = float(np.max(np.diff(t_sec))) if len(t_sec) > 1 else 0.0
    meta.update(n_wraps=n_wraps, n_dups=int((~uniq).sum()), duration_s=dur,
                pkt_rate_hz=float(len(ts) / dur) if dur > 0 else 0.0, max_gap_ms=max_gap * 1000)
    if dur < MIN_DUR_S:
        meta["exclusion_reason"] = f"short_duration: {dur:.3f}s < {MIN_DUR_S}s"
        return None, meta
    if max_gap > MAX_GAP_S:
        meta["exclusion_reason"] = f"packet_gap: max {max_gap * 1000:.1f}ms > {MAX_GAP_S * 1000:.0f}ms"
        return None, meta

    ant1, ant2 = csi[:, :, 0, 0], csi[:, :, 1, 0]   # RF chains 0 and 1 = antennas 1 and 2 of r1
    if hampel_mode == "antenna_complex":
        ant1, ant2 = hampel_filter_complex(ant1), hampel_filter_complex(ant2)
    R = compute_csi_ratio(ant1, ant2)
    if hampel_mode == "ratio":
        R = hampel_filter_complex(R)
    R = resample_uniform(t_sec, R)
    if not np.isfinite(R).all():
        meta["exclusion_reason"] = "non_finite_ratio"
        return None, meta
    return R, meta


# --------------------------------------------------------------------------- feature maps
def _sos(kind: str):
    if kind == "band":
        return scipy.signal.butter(4, BAND_HZ, btype="bandpass", fs=FS, output="sos")
    if kind == "hp":
        return scipy.signal.butter(2, BAND_HZ[0], btype="highpass", fs=FS, output="sos")
    if kind == "lp":
        return scipy.signal.butter(4, BAND_HZ[1], btype="lowpass", fs=FS, output="sos")
    raise ValueError(kind)


def _filtfilt(kind: str, x: np.ndarray) -> np.ndarray:
    """Zero-phase Butterworth along time (axis 0). Works on complex input (real coefficients)."""
    return scipy.signal.sosfiltfilt(_sos(kind), x, axis=0)


def phase_map(R: np.ndarray, variant: str) -> np.ndarray:
    """(T, 30) complex ratio -> (30, T') real phase map.
      unwrap    : unwrapped angle(R) minus its per-subcarrier mean (Phase-2 definition)
      unwrap_hp : the same, then zero-phase 2 Hz high-pass (removes drift and unwrap offsets)
      rel       : angle(R * conj(mean_t R)), phase relative to the static vector, in (-pi, pi]; no unwrapping
      diff      : angle(Rs[t] * conj(Rs[t-1])) with Rs = R low-passed at 60 Hz; phase rate, no unwrapping
    """
    if variant in ("unwrap", "unwrap_hp"):
        ph = np.unwrap(np.angle(R), axis=0)
        ph = ph - ph.mean(axis=0, keepdims=True)
        if variant == "unwrap_hp":
            ph = _filtfilt("hp", ph)
    elif variant == "rel":
        ph = np.angle(R * np.conj(R.mean(axis=0, keepdims=True)))
    elif variant == "diff":
        Rs = _filtfilt("lp", R)
        ph = np.angle(Rs[1:] * np.conj(Rs[:-1]))
    else:
        raise ValueError(f"unknown phase variant {variant}")
    return ph.T.astype(np.float32)


def stft_power(x: np.ndarray) -> np.ndarray:
    """x: (T, S) complex. Hann 64, hop 16, n_fft 256, reflect-padded so frames are centred.
    Returns two-sided power restricted to |f| <= 60 Hz, ascending frequency: (S, F, T')."""
    pad = STFT_WIN // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(xp, STFT_WIN, axis=0)[::STFT_HOP]   # (T', S, win)
    spec = np.fft.fft(frames * scipy.signal.get_window("hann", STFT_WIN), n=STFT_NFFT, axis=-1)
    spec = np.fft.fftshift(spec, axes=-1)
    freqs = np.fft.fftshift(np.fft.fftfreq(STFT_NFFT, d=1.0 / FS))
    keep = np.abs(freqs) <= DOPPLER_MAX_HZ
    return (np.abs(spec[..., keep]) ** 2).transpose(1, 2, 0)


def doppler_map(R: np.ndarray, variant: str) -> np.ndarray:
    """(T, 30) complex ratio -> (F, T') real Doppler map. Variant = <static removal>_<aggregation>_<scale>:
      mean | bp   : subtract the per-subcarrier temporal mean | zero-phase Butterworth band-pass 2-60 Hz
      avg  | pca  : STFT power averaged over 30 subcarriers | STFT power of the first complex principal component
      log1p | db  : log(1 + p) | 10*log10(p / p99) clipped to [-40, 0] dB (per-trial reference, scale-free)
    """
    static, agg, scale = variant.split("_")
    if static == "mean":
        X = R - R.mean(axis=0, keepdims=True)
    elif static == "bp":
        X = _filtfilt("band", R)
    else:
        raise ValueError(variant)
    if agg == "avg":
        p = stft_power(X).mean(axis=0)
    elif agg == "pca":
        Xc = X - X.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(Xc, full_matrices=False)
        p = stft_power((Xc @ np.conj(vh[0]))[:, None])[0]
    else:
        raise ValueError(variant)
    if scale == "log1p":
        out = np.log1p(p)
    elif scale == "db":
        ref = np.percentile(p, 99.0)
        if not ref > 0:
            raise FloatingPointError("Doppler power reference is not positive")
        out = np.clip(10.0 * np.log10(np.maximum(p, 1e-30) / ref), DB_FLOOR, 0.0)
    else:
        raise ValueError(variant)
    return out.astype(np.float32)


def resize_map(m: np.ndarray, shape: Tuple[int, int] = MAP_SHAPE) -> np.ndarray:
    """Anti-aliased bilinear resize of a 2-D map."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None, None]
    return F.interpolate(t, size=shape, mode="bilinear", align_corners=False, antialias=True)[0, 0].numpy()


def to_fp16(x: np.ndarray) -> Optional[np.ndarray]:
    if not np.isfinite(x).all() or np.abs(x).max() > 6e4:
        return None
    return x.astype(np.float16)


def preprocess_trial(csi, timestamps, cfg: Optional[Dict[str, str]] = None):
    """Locked end-to-end pipeline for one trial -> ((2, 64, 128) float16 [phase, Doppler] or None, meta)."""
    cfg = cfg or LOCKED
    t0 = time.perf_counter()
    R, meta = ratio_200hz(csi, timestamps, cfg["hampel_mode"])
    if R is None:
        return None, meta
    x = np.stack([resize_map(phase_map(R, cfg["phase_variant"])),
                  resize_map(doppler_map(R, cfg["doppler_variant"]))])
    x16 = to_fp16(x)
    if x16 is None:
        meta["exclusion_reason"] = "non_finite_or_fp16_overflow"
    meta["feature_ms"] = (time.perf_counter() - t0) * 1000
    return x16, meta


def run_unit_tests() -> None:
    # timestamp wrap + reorder
    t = (np.arange(3000, dtype=np.int64) * 1000 + (1 << 32) - 1_500_000) % (1 << 32)
    t[2000], t[2001] = t[2001], t[2000]
    tu, nw = unwrap_timestamps(t.astype(np.uint32))
    assert nw == 1 and np.all(np.diff(np.sort(tu)) == 1000)
    # Hampel: isolated spike removed, clean ramp untouched
    sig = np.linspace(0, 1, 50)[:, None].repeat(3, 1)
    spk = sig.copy(); spk[25, 1] = 10.0
    assert abs(hampel_filter(spk)[25, 1] - sig[25, 1]) < 0.05 and np.allclose(hampel_filter(sig), sig)
    # Doppler: a +20 Hz tone on a static offset must peak at +20 Hz; the band-pass must remove the 0 Hz line
    tt = np.arange(400) / FS
    R = (1.0 + 0.2 * np.exp(2j * np.pi * 20.0 * tt))[:, None].repeat(30, 1) + 0.3 * tt[:, None]
    freqs = np.fft.fftshift(np.fft.fftfreq(STFT_NFFT, d=1.0 / FS))
    freqs = freqs[np.abs(freqs) <= DOPPLER_MAX_HZ]
    for v in DOPPLER_VARIANTS:
        d = doppler_map(R, v)
        assert d.shape[0] == len(freqs) and np.isfinite(d).all()
        prof = d[:, 3:-3].mean(axis=1)
        assert abs(freqs[int(np.argmax(prof))] - 20.0) < 1.0, (v, freqs[int(np.argmax(prof))])
    i0 = int(np.argmin(np.abs(freqs)))
    lin = lambda v: np.expm1(doppler_map(R, v))[i0, 3:-3].mean()
    assert lin("bp_avg_log1p") < 0.05 * lin("mean_avg_log1p"), "band-pass did not suppress the drift at 0 Hz"
    # phase variants: shapes and finiteness; 'rel' is bounded and has no 2*pi steps
    for v in PHASE_VARIANTS:
        p = phase_map(R, v)
        assert p.shape[0] == 30 and np.isfinite(p).all()
    assert np.abs(phase_map(R, "rel")).max() <= np.pi + 1e-6
    assert resize_map(doppler_map(R, "bp_avg_db")).shape == MAP_SHAPE
    assert set(LOCKED) == {"hampel_mode", "phase_variant", "doppler_variant"}
    assert LOCKED["hampel_mode"] in HAMPEL_MODES and LOCKED["phase_variant"] in PHASE_VARIANTS \
        and LOCKED["doppler_variant"] in DOPPLER_VARIANTS
    print("preprocess unit tests PASSED")


if __name__ == "__main__":
    run_unit_tests()
