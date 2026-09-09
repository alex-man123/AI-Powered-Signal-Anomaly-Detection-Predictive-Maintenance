"""TASK 5.2 — frequency-domain vibration features.

Six pure, independent functions. Unlike TASK 5.1's time-domain features (which take
a raw signal), these all operate on an ALREADY-COMPUTED spectral representation --
`(frequencies, power)` -- since TASK 4.1 (`compute_fft`) and TASK 4.2
(`compute_welch_psd`) already own the responsibility of turning a signal into a
spectrum, and this module must not duplicate that (per this task's own "Reutilizare"
instruction).

Spectral convention (read before interpreting any result) -- MUST be documented since
this task explicitly warns against silently mixing FFT magnitude / FFT power / PSD:

- All five energy/distribution-based features (`spectral_centroid`,
  `spectral_bandwidth`, `spectral_energy`, `spectral_entropy`, `band_energy`) are
  documented and tested against **Welch PSD** (TASK 4.2's `compute_welch_psd`) as the
  canonical spectral input -- `power` is expected to be a power spectral density
  (non-negative, units signal^2/Hz), not raw FFT magnitude. Two reasons: (1) PSD is
  already a power/energy quantity, matching what these formulas are physically
  defined over, whereas FFT magnitude is an amplitude; (2) blueprint.md section 13
  itself prefers Welch PSD over simple FFT for spectral estimation precisely because
  it is "mai robusta la zgomot" -- the more appropriate choice to feed feature
  extraction from real (noisy) vibration data, a difference TASK 4.2's own AC1 test
  already measured directly (`cv_welch` visibly lower than `cv_fft`).
  Nothing here recomputes a PSD or FFT internally -- the caller runs TASK 4.1/4.2
  first and passes the result in.
- `dominant_frequency` is the one exception: it is a thin, literal reuse of TASK
  4.1's own `dominant_frequency(freqs, magnitude)` (an argmax over whatever spectral
  array is passed), not reimplemented, and therefore agnostic to which spectral
  representation is used -- but in practice this module's own tests feed it the same
  Welch PSD as every other feature here, for a single consistent spectral source per
  signal.

`spectral_energy` is defined as the **area under the power spectral density curve**
(`numpy.trapezoid(power, frequencies)`, i.e. the numerical integral of PSD over
frequency) -- NOT a bare `sum(power)`. This is the definition consistent with PSD's
own units (power *density*, per Hz) and with Rayleigh/Parseval's energy theorem
(integrating a signal's PSD over all frequencies recovers its average power); it is
also empirically near-invariant to the `nperseg` choice used to compute the PSD
(measured: a 2.0 V-amplitude sinusoid's spectral energy stayed within 0.01% of the
theoretical A^2/2 = 2.0 across nperseg in {256, 512, 1024, 2048}), whereas a bare
`sum(power)` would scale with the number of frequency bins and therefore with
`nperseg` -- confirming the trapezoidal-integral definition is the physically
meaningful, implementation-parameter-independent one. `band_energy` uses the exact
same area-under-the-curve definition, restricted to a frequency sub-range.

Not all six features need a distinct implementation to guard against divide-by-zero
identically: `spectral_centroid`/`spectral_bandwidth`/`spectral_entropy` all raise
`FrequencyFeatureError` when total spectral power is zero (an all-zero signal's
spectrum), since each would otherwise divide by zero or take log(0) of an undefined
distribution -- matching the divide-by-zero convention already established by TASK
3.1 (`PreprocessingError`) and TASK 5.1 (`FeatureError`).
"""

from __future__ import annotations

import numpy as np

from app.signal_processing.fft import dominant_frequency as _fft_dominant_frequency


class FrequencyFeatureError(ValueError):
    """Raised for invalid frequency-domain feature input (empty/mismatched
    frequencies-power arrays, negative power, an invalid frequency band) or a
    mathematically undefined result (total spectral power is zero). Never silently
    corrected."""


def _validate_spectrum(frequencies: np.ndarray, power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.asarray(frequencies, dtype=float)
    power = np.asarray(power, dtype=float)

    if frequencies.size == 0 or power.size == 0:
        raise FrequencyFeatureError("frequencies and power must not be empty")
    if frequencies.shape != power.shape:
        raise FrequencyFeatureError(
            f"frequencies and power must have the same shape, got {frequencies.shape} and {power.shape}"
        )
    if np.any(power < 0):
        raise FrequencyFeatureError(
            "power must be non-negative -- this module expects a power/energy spectral "
            "representation (e.g. a PSD), never a signed value"
        )
    return frequencies, power


def dominant_frequency(frequencies: np.ndarray, power: np.ndarray) -> float:
    """Frequency of the largest-power spectral bin.

    This is TASK 4.1's own `dominant_frequency` (`app.signal_processing.fft`),
    called directly and unmodified -- no new estimator, no interpolation/peak
    fitting/zero crossing/sub-bin estimation, per this task's explicit instruction.

    Formula:
        dominant_frequency = frequencies[argmax(power)]

    Args:
        frequencies: spectral frequency axis, Hz.
        power: corresponding spectral power (or magnitude) values, same length.

    Returns:
        The frequency (Hz) of the largest-power bin, at spectral-resolution
        precision (`fs / nperseg` for a Welch PSD, `fs / N` for a raw FFT) -- no
        finer than that.

    Physical interpretation:
        The single most energetic frequency component of the signal.

    Fault relevance:
        A shift in dominant frequency (e.g. away from the shaft's known rotation
        frequency or its harmonics) can indicate a developing fault whose energy is
        concentrated at a specific frequency.

    Raises:
        FrequencyFeatureError: not raised here -- this delegates to TASK 4.1's own
            `dominant_frequency`, which raises `app.signal_processing.fft.FFTError`
            (a different exception type) for mismatched-length or empty input, since
            this literally IS that function, not a reimplementation.
    """
    return _fft_dominant_frequency(frequencies, power)


def spectral_centroid(frequencies: np.ndarray, power: np.ndarray) -> float:
    """Spectral centroid: the power-weighted "center of mass" frequency.

    Formula:
        centroid = sum(f_i * P_i) / sum(P_i)

    Args:
        frequencies: spectral frequency axis, Hz.
        power: non-negative spectral power (PSD convention -- see module docstring).

    Physical interpretation:
        Represents the center of mass of the spectral energy distribution -- for
        energy concentrated near a single frequency X, the centroid is close to X.

    Fault relevance:
        Shifts in spectral centroid can indicate changes in vibration energy
        distribution associated with changing machine operating conditions or
        developing faults, even when the raw dominant-frequency bin does not move.

    Raises:
        FrequencyFeatureError: if `frequencies`/`power` are empty, mismatched in
            length, contain negative power, or total power is zero (the centroid
            would divide by zero).
    """
    frequencies, power = _validate_spectrum(frequencies, power)

    total_power = float(np.sum(power))
    if total_power == 0:
        raise FrequencyFeatureError(
            "Cannot compute spectral centroid: total spectral power is zero (silent/all-zero spectrum)."
        )

    return float(np.sum(frequencies * power) / total_power)


def spectral_bandwidth(frequencies: np.ndarray, power: np.ndarray) -> float:
    """Spectral bandwidth: the power-weighted standard deviation of frequency around
    the spectral centroid.

    Formula:
        bandwidth = sqrt( sum(P_i * (f_i - centroid)^2) / sum(P_i) )

    Args:
        frequencies: spectral frequency axis, Hz.
        power: non-negative spectral power (PSD convention -- see module docstring).

    Physical interpretation:
        A small bandwidth means spectral energy is concentrated in a narrow
        frequency range (e.g. a pure tone); a large bandwidth means it is spread
        across a wide range (e.g. broadband noise).

    Fault relevance:
        Some fault mechanisms broaden the vibration spectrum around a base frequency
        (energy spreads into sidebands/harmonics) -- an increasing bandwidth can flag
        this even when the centroid itself barely moves.

    Raises:
        FrequencyFeatureError: if `frequencies`/`power` are empty, mismatched in
            length, contain negative power, or total power is zero (the same
            condition under which `spectral_centroid` -- called internally -- would
            raise).
    """
    frequencies, power = _validate_spectrum(frequencies, power)

    total_power = float(np.sum(power))
    centroid = spectral_centroid(frequencies, power)

    variance = float(np.sum(power * (frequencies - centroid) ** 2) / total_power)
    return float(np.sqrt(variance))


def spectral_energy(frequencies: np.ndarray, power: np.ndarray) -> float:
    """Spectral energy: the area under the power spectral density curve.

    Formula:
        energy = integral(P(f) df) ~= numpy.trapezoid(power, frequencies)

    Args:
        frequencies: spectral frequency axis, Hz (expected to come from a PSD --
            see module docstring).
        power: non-negative power spectral density values, same length.

    Physical interpretation:
        By Rayleigh/Parseval's energy theorem, integrating a signal's PSD over all
        frequencies recovers its average power -- e.g. for a pure sinusoid of
        amplitude A, this evaluates to ~= A^2/2 (measured: within 0.01% of the
        theoretical value for a clean synthetic sinusoid). This is why the
        trapezoidal integral is used rather than a bare `sum(power)`, which has no
        such physical meaning and, empirically, scales with the number of frequency
        bins (i.e. with the PSD's own `nperseg` parameter) rather than staying
        approximately constant.

    Fault relevance:
        Overall increase in spectral energy is a broad indicator of increased
        vibration severity, complementary to time-domain RMS (TASK 5.1) but computed
        entirely in the frequency domain.

    Raises:
        FrequencyFeatureError: if `frequencies`/`power` are empty, mismatched in
            length, or contain negative power.
    """
    frequencies, power = _validate_spectrum(frequencies, power)
    return float(np.trapezoid(power, frequencies))


def spectral_entropy(frequencies: np.ndarray, power: np.ndarray) -> float:
    """Normalized spectral entropy: how concentrated vs. spread out the spectral
    power distribution is.

    Formula:
        p_i = P_i / sum(P_j)                     -- normalized spectral distribution
        H = -sum(p_i * log(p_i)), over p_i > 0    -- Shannon entropy (log(0) skipped)
        H_normalized = H / log(N)                 -- N = number of frequency bins

    `frequencies` is accepted for API symmetry with the other five features (all of
    which take `(frequencies, power)`) but is not used in the formula itself --
    spectral entropy depends only on the shape of the power distribution across
    bins, not on their frequency values.

    Args:
        frequencies: spectral frequency axis, Hz (unused in the computation itself).
        power: non-negative spectral power (PSD convention -- see module docstring).

    Returns:
        A value in `[0, 1]`: ~=0 means spectral power is concentrated in very few
        bins (e.g. a pure sinusoid); ~=1 means it is spread roughly uniformly across
        all bins (e.g. white noise). Measured: a clean 100 Hz sinusoid's normalized
        entropy ~= 0.156; fixed-seed white Gaussian noise's ~= 0.993, over the same
        spectral resolution -- a clear, non-marginal separation.

    Physical interpretation:
        Low entropy = energy concentrated at specific frequencies (tonal); high
        entropy = energy spread broadly (noise-like).

    Fault relevance:
        Some faults introduce broadband energy (increasing entropy) while others
        concentrate energy at specific harmonics (decreasing entropy) -- entropy
        complements spectral bandwidth as a distribution-shape indicator.

    Raises:
        FrequencyFeatureError: if `frequencies`/`power` are empty, mismatched in
            length, contain negative power, or total power is zero (no valid
            probability distribution can be formed).
    """
    frequencies, power = _validate_spectrum(frequencies, power)

    total_power = float(np.sum(power))
    if total_power == 0:
        raise FrequencyFeatureError(
            "Cannot compute spectral entropy: total spectral power is zero (silent/all-zero spectrum)."
        )

    num_bins = power.size
    if num_bins == 1:
        # A single-bin spectrum is, by definition, maximally concentrated (entropy
        # 0) -- and log(1) == 0 would otherwise make the normalization step below
        # divide by zero.
        return 0.0

    probabilities = power / total_power
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))

    return entropy / np.log(num_bins)


def band_energy(frequencies: np.ndarray, power: np.ndarray, low_freq: float, high_freq: float) -> float:
    """Spectral energy restricted to a `[low_freq, high_freq]` frequency band.

    Formula:
        band_energy = integral(P(f) df) for f in [low_freq, high_freq]
                    ~= numpy.trapezoid(power[mask], frequencies[mask])

    Same area-under-the-curve definition as `spectral_energy`, restricted to the
    sub-band via a boolean mask on `frequencies` (inclusive of both endpoints).

    Args:
        frequencies: spectral frequency axis, Hz.
        power: non-negative spectral power (PSD convention -- see module docstring).
        low_freq: band lower bound, Hz, must be `>= 0`.
        high_freq: band upper bound, Hz, must be `> low_freq` and
            `<= frequencies.max()` (this module has no independent knowledge of the
            sampling rate/Nyquist frequency -- it trusts that `frequencies`, as
            produced by TASK 4.1/4.2, already stops at Nyquist, and treats
            `frequencies.max()` as that bound).

    Returns:
        The energy contained in the band. A band containing fewer than 2 frequency
        bins (too narrow relative to spectral resolution, or landing entirely
        between two bins) returns `0.0` -- an honest "no measured energy in this
        band" answer, not an error, since a trapezoidal integral is undefined over
        fewer than 2 points.

    Physical interpretation:
        Energy specifically attributable to a known frequency range (e.g. around a
        shaft's rotation frequency or a specific fault harmonic), as opposed to
        `spectral_energy`'s whole-spectrum total.

    Fault relevance:
        Many faults have known characteristic frequencies (e.g. bearing defect
        frequencies, gear mesh frequencies) -- tracking energy in those specific
        bands is more targeted than whole-spectrum energy or RMS.

    Raises:
        FrequencyFeatureError: if `frequencies`/`power` are empty, mismatched in
            length, contain negative power, `low_freq < 0`, `high_freq <= low_freq`,
            or `high_freq > frequencies.max()`.
    """
    frequencies, power = _validate_spectrum(frequencies, power)

    if low_freq < 0:
        raise FrequencyFeatureError(f"low_freq must be >= 0, got {low_freq}")
    if high_freq <= low_freq:
        raise FrequencyFeatureError(f"high_freq ({high_freq}) must be > low_freq ({low_freq})")
    max_freq = float(frequencies.max())
    if high_freq > max_freq:
        raise FrequencyFeatureError(
            f"high_freq ({high_freq}) must not exceed the highest available frequency ({max_freq}), "
            "which this module treats as the Nyquist bound"
        )

    mask = (frequencies >= low_freq) & (frequencies <= high_freq)
    if mask.sum() < 2:
        return 0.0

    return float(np.trapezoid(power[mask], frequencies[mask]))
