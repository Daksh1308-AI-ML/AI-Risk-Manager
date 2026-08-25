from __future__ import annotations

import numpy as np
from river.drift import ADWIN, PageHinkley


class DriftDetector:
    """Multi-method drift detection system.

    Combines ADWIN, Page-Hinkley, and Population Stability Index (PSI)
    to detect concept drift in model predictions and feature distributions.

    Parameters
    ----------
    warning_level : float
        Threshold for warning alerts (0-1 range). Default 0.1.
    drift_level : float
        Threshold for drift alerts (0-1 range). Default 0.2.
    """

    def __init__(self, warning_level: float = 0.1, drift_level: float = 0.2) -> None:
        self.warning_level = warning_level
        self.drift_level = drift_level

        self.adwin = ADWIN(delta=0.002)
        self.page_hinkley = PageHinkley(threshold=50, delta=0.005)

        self._adwin_value: float = 0.0
        self._page_hinkley_drift: bool = False
        self._page_hinkley_value: float = 0.0
        self._observation_count: int = 0
        self._running_sum: float = 0.0

    def update(self, value: float) -> dict:
        """Update all detectors with a new observation.

        Parameters
        ----------
        value : float
            New observation (e.g., prediction error, feature value).

        Returns
        -------
        dict
            Keys: adwin_status, adwin_value, page_hinkley_value,
            page_hinkley_drift.
        """
        self.adwin.update(value)
        self.page_hinkley.update(value)

        self._observation_count += 1
        self._running_sum += value

        self._adwin_value = float(self.adwin.estimation)
        self._page_hinkley_drift = self.page_hinkley.drift_detected
        self._page_hinkley_value = self._running_sum / self._observation_count

        adwin_status = "stable"
        if self.adwin.drift_detected:
            adwin_status = "drift"
        elif self._adwin_value > self.warning_level:
            adwin_status = "warning"

        return {
            "adwin_status": adwin_status,
            "adwin_value": self._adwin_value,
            "page_hinkley_value": self._page_hinkley_value,
            "page_hinkley_drift": self._page_hinkley_drift,
        }

    def detect_psi(
        self, expected: np.ndarray, actual: np.ndarray, bins: int = 10
    ) -> dict:
        """Calculate Population Stability Index.

        Parameters
        ----------
        expected : np.ndarray
            Reference distribution (e.g., training predictions).
        actual : np.ndarray
            Current distribution (e.g., recent predictions).
        bins : int
            Number of bins for histogram. Default 10.

        Returns
        -------
        dict
            Keys: psi_value, drift_detected, psi_threshold.
        """
        psi_value = self._calculate_psi(expected, actual, bins)
        drift_detected = psi_value >= self.drift_level

        return {
            "psi_value": psi_value,
            "drift_detected": drift_detected,
            "psi_threshold": self.drift_level,
        }

    def _calculate_psi(
        self, expected: np.ndarray, actual: np.ndarray, bins: int = 10
    ) -> float:
        """Calculate PSI between two distributions.

        PSI < 0.1: No significant change.
        0.1 <= PSI < 0.2: Moderate change.
        PSI >= 0.2: Significant drift.

        Parameters
        ----------
        expected : np.ndarray
            Reference distribution.
        actual : np.ndarray
            Current distribution.
        bins : int
            Number of bins.

        Returns
        -------
        float
            PSI value.
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        exp_var = np.var(expected)
        act_var = np.var(actual)
        if exp_var == 0 and act_var == 0:
            return 0.0

        min_val = min(float(expected.min()), float(actual.min()))
        max_val = max(float(expected.max()), float(actual.max()))

        if min_val == max_val:
            return 0.0

        edges = np.linspace(min_val, max_val, bins + 1)

        exp_counts, _ = np.histogram(expected, bins=edges)
        act_counts, _ = np.histogram(actual, bins=edges)

        eps = 1e-6
        exp_pct = (exp_counts + eps) / (exp_counts.sum() + eps * bins)
        act_pct = (act_counts + eps) / (act_counts.sum() + eps * bins)

        psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
        return psi

    def get_status(self) -> dict:
        """Get current status of all detectors.

        Returns
        -------
        dict
            Keys: overall_drift_detected, adwin_drift, adwin_warning,
            page_hinkley_drift, adwin_value, page_hinkley_value.
        """
        adwin_drift = self.adwin.drift_detected
        adwin_warning = self._adwin_value > self.warning_level
        ph_drift = self._page_hinkley_drift

        return {
            "overall_drift_detected": adwin_drift or ph_drift,
            "adwin_drift": adwin_drift,
            "adwin_warning": adwin_warning,
            "page_hinkley_drift": ph_drift,
            "adwin_value": self._adwin_value,
            "page_hinkley_value": self._page_hinkley_value,
        }

    def reset(self) -> None:
        """Reset all detectors to initial state."""
        self.adwin = ADWIN(delta=0.002)
        self.page_hinkley = PageHinkley(threshold=50, delta=0.005)
        self._adwin_value = 0.0
        self._page_hinkley_drift = False
        self._page_hinkley_value = 0.0
        self._observation_count = 0
        self._running_sum = 0.0


if __name__ == "__main__":
    print("=== Drift Detection Demo ===\n")

    detector = DriftDetector(warning_level=0.1, drift_level=0.2)

    print("--- Simulating stable stream ---")
    rng = np.random.default_rng(42)
    stable_values = rng.normal(0, 1, 100)
    for i, v in enumerate(stable_values):
        result = detector.update(float(v))
        if result["adwin_status"] != "stable":
            print(f"  Step {i}: {result}")

    status = detector.get_status()
    print(f"  Status after stable stream: {status}\n")

    print("--- Simulating drift (mean shift) ---")
    drifted_values = rng.normal(3, 1, 100)
    drift_step = None
    for i, v in enumerate(drifted_values):
        result = detector.update(float(v))
        if result["adwin_status"] == "drift" and drift_step is None:
            drift_step = i
            print(f"  Drift detected at step {i}")
            print(f"    adwin_status: {result['adwin_status']}")
            print(f"    page_hinkley_drift: {result['page_hinkley_drift']}")

    if drift_step is None:
        print("  No drift detected (try adjusting thresholds)")
    print()

    print("--- Testing PSI ---")
    expected = rng.normal(0, 1, 500)
    actual_stable = rng.normal(0, 1, 500)
    actual_drifted = rng.normal(2, 1.5, 500)

    psi_stable = detector.detect_psi(expected, actual_stable)
    psi_drifted = detector.detect_psi(expected, actual_drifted)

    print(f"  PSI (no drift):     {psi_stable['psi_value']:.4f} -> drift={psi_stable['drift_detected']}")
    print(f"  PSI (with drift):   {psi_drifted['psi_value']:.4f} -> drift={psi_drifted['drift_detected']}")
    print()

    print("--- Edge cases ---")
    print(f"  Empty arrays: {detector.detect_psi(np.array([]), np.array([]))['psi_value']:.4f}")
    print(f"  Zero variance: {detector.detect_psi(np.ones(100), np.ones(100))['psi_value']:.4f}")
    print()

    detector.reset()
    print("--- After reset ---")
    print(f"  Status: {detector.get_status()}")
