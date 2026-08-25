"""Feature engineering pipeline for chargeback prediction.

Transforms raw transaction data into 20 model-ready features covering
velocity, device trust, geographic anomalies, account history, temporal
patterns, and amount distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical feature list
# ---------------------------------------------------------------------------

FEATURES: list[str] = [
    # Velocity features
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "amount_sum_24h",
    "avg_amount_diff",
    # Device features
    "device_trust_score",
    "is_new_device",
    "device_age_days",
    # Geographic features
    "geo_velocity",
    "is_new_address",
    "ip_country_match",
    # Account features
    "account_age_days",
    "past_disputes",
    "dispute_rate",
    "account_activity",
    # Temporal features
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    # Amount features
    "amount_percentile",
]

TARGET_COLS: list[str] = ["chargeback_label", "fraud_type", "transaction_id"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, np.dtype] = {
    "txn_count_1h": np.int64,
    "txn_count_24h": np.int64,
    "txn_count_7d": np.int64,
    "amount_sum_24h": np.float64,
    "avg_amount_diff": np.float64,
    "device_trust_score": np.float64,
    "is_new_device": np.float64,
    "device_age_days": np.float64,
    "geo_velocity": np.float64,
    "is_new_address": np.float64,
    "ip_country_match": np.float64,
    "account_age_days": np.float64,
    "past_disputes": np.float64,
    "dispute_rate": np.float64,
    "account_activity": np.float64,
    "hour_of_day": np.float64,
    "day_of_week": np.float64,
    "is_weekend": np.float64,
    "is_night": np.float64,
    "amount_percentile": np.float64,
}


class FeatureEngine:
    """Feature engineering pipeline for chargeback prediction.

    Learns customer-, device-, and amount-level statistics from training data
    during ``fit()`` and applies deterministic transforms during
    ``transform()``.

    Parameters
    ----------
    feature_names:
        Ordered list of feature column names the output DataFrame must contain.
        Defaults to the module-level ``FEATURES`` constant.
    """

    def __init__(self, feature_names: list[str] = FEATURES) -> None:
        self.feature_names = list(feature_names)

        # Fitted state (populated by ``fit``)
        self._customer_mean_amount: dict[str, float] = {}
        self._customer_txn_count: dict[str, int] = {}
        self._device_total: dict[str, int] = {}
        self._device_non_chargeback: dict[str, int] = {}
        self._device_first_seen: dict[str, pd.Timestamp] = {}
        self._amount_percentiles: np.ndarray | None = None
        self._global_mean_amount: float = 0.0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> FeatureEngine:
        """Learn customer-level and device-level stats from training data.

        Parameters
        ----------
        df:
            Raw transaction DataFrame containing at minimum the columns
            listed in ``TransactionSchema``.

        Returns
        -------
        FeatureEngine
            ``self``, to allow method chaining.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

        # --- Customer-level aggregates ---
        customer_groups = df.groupby("customer_id")
        self._customer_mean_amount = customer_groups["amount"].mean().to_dict()
        self._customer_txn_count = customer_groups["transaction_id"].count().to_dict()

        # --- Device-level aggregates ---
        device_groups = df.groupby("device_fingerprint")
        self._device_total = device_groups["transaction_id"].count().to_dict()
        chargebacks_per_device = df.groupby("device_fingerprint")["chargeback_label"].sum().to_dict()
        self._device_non_chargeback = {
            dev: int(self._device_total[dev] - chargebacks_per_device.get(dev, 0))
            for dev in self._device_total
        }
        self._device_first_seen = device_groups["timestamp"].min().to_dict()

        # --- Amount distribution for percentile ranking ---
        self._amount_percentiles = np.sort(df["amount"].values)
        self._global_mean_amount = float(df["amount"].mean())

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw transaction data.

        Must call ``fit()`` first (or use ``fit_transform()``).

        Parameters
        ----------
        df:
            Raw transaction DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with exactly the 20 feature columns, plus target
            columns ``chargeback_label``, ``fraud_type``, and
            ``transaction_id`` for convenience.
        """
        if not self._fitted:
            raise RuntimeError("FeatureEngine has not been fitted. Call fit() first.")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
        n = len(df)

        # --- Vectorised features (no row loop) ---
        ts = df["timestamp"]
        out: dict[str, np.ndarray] = {}

        # Temporal
        out["hour_of_day"] = ts.dt.hour.values.astype(np.float64)
        out["day_of_week"] = ts.dt.dayofweek.values.astype(np.float64)
        out["is_weekend"] = np.where(ts.dt.dayofweek >= 5, 1.0, 0.0)
        out["is_night"] = np.where((ts.dt.hour >= 22) | (ts.dt.hour < 6), 1.0, 0.0)

        # Passthrough
        out["is_new_device"] = df["is_new_device"].astype(np.float64).values
        out["is_new_address"] = df["is_new_address"].astype(np.float64).values
        out["account_age_days"] = df["account_age_days"].astype(np.float64).values
        out["past_disputes"] = df["past_disputes"].astype(np.float64).values

        # Account features
        acct_months = df["account_age_days"].values.astype(np.float64) / 30.0 + 1.0
        out["dispute_rate"] = df["past_disputes"].values.astype(np.float64) / acct_months

        cust_counts = df["customer_id"].map(self._customer_txn_count).fillna(0).values.astype(np.float64)
        acct_weeks = np.maximum(df["account_age_days"].values.astype(np.float64) / 7, 1)
        out["account_activity"] = cust_counts / acct_weeks

        # Device features
        device_total = df["device_fingerprint"].map(self._device_total).fillna(0).values
        device_ncb = df["device_fingerprint"].map(self._device_non_chargeback).fillna(0).values
        out["device_trust_score"] = np.where(device_total > 0, device_ncb / device_total, 0.5)

        device_first_s = df["device_fingerprint"].map(
            {k: v.timestamp() for k, v in self._device_first_seen.items()}
        ).fillna(ts.min().timestamp()).values.astype(np.float64)
        out["device_age_days"] = (ts.values.astype(np.int64) / 1e6 - device_first_s) / 86400.0

        # ip_country_match (always 1.0 for Indian synthetic data)
        out["ip_country_match"] = np.ones(n, dtype=np.float64)

        # Avg amount diff
        cust_means = df["customer_id"].map(self._customer_mean_amount).fillna(self._global_mean_amount).values
        out["avg_amount_diff"] = df["amount"].values.astype(np.float64) - cust_means

        # Amount percentile
        if self._amount_percentiles is not None and len(self._amount_percentiles) > 0:
            out["amount_percentile"] = np.searchsorted(
                self._amount_percentiles, df["amount"].values
            ) / len(self._amount_percentiles)
        else:
            out["amount_percentile"] = np.full(n, 0.5)

        # --- Velocity features (per-customer, sequential within groups) ---
        # Use groupby transform for efficiency
        velocity_1h = np.zeros(n, dtype=np.int64)
        velocity_24h = np.zeros(n, dtype=np.int64)
        velocity_7d = np.zeros(n, dtype=np.int64)
        amount_sum_24h = np.zeros(n, dtype=np.float64)
        geo_velocity = np.zeros(n, dtype=np.float64)

        ts_vals = ts.values.astype(np.int64)  # microsecond epoch
        amt_vals = df["amount"].values.astype(np.float64)
        ip_vals = df["ip_address"].values
        cid_vals = df["customer_id"].values

        one_h_us = np.int64(3_600 * 1_000_000)
        one_d_us = np.int64(86_400 * 1_000_000)
        seven_d_us = np.int64(7 * 86_400 * 1_000_000)

        # Group boundaries (data already sorted by customer_id, timestamp)
        group_starts: dict[str, int] = {}
        prev_cid = None
        for i in range(n):
            c = cid_vals[i]
            if c != prev_cid:
                group_starts[c] = i
                prev_cid = c

        # Within each customer group, compute velocity features
        # This is O(total_txn) since each row is visited once per group
        for cid, start_idx in group_starts.items():
            # Find group end
            end_idx = start_idx + 1
            while end_idx < n and cid_vals[end_idx] == cid:
                end_idx += 1

            grp_size = end_idx - start_idx
            if grp_size <= 1:
                # First transaction for this customer — all velocities are 0
                # except self-count
                velocity_1h[start_idx] = 1
                velocity_24h[start_idx] = 1
                velocity_7d[start_idx] = 1
                continue

            # Sliding window counters
            cumulative_count = 0
            cumulative_amount = 0.0
            head_1h = start_idx
            head_24h = start_idx
            head_7d = start_idx

            for i in range(start_idx, end_idx):
                cur_ts = ts_vals[i]
                cur_amt = amt_vals[i]
                cumulative_count += 1
                cumulative_amount += cur_amt

                # Advance heads past expiry
                while head_1h < i and (cur_ts - ts_vals[head_1h]) > one_h_us:
                    head_1h += 1
                while head_24h < i and (cur_ts - ts_vals[head_24h]) > one_d_us:
                    head_24h += 1
                while head_7d < i and (cur_ts - ts_vals[head_7d]) > seven_d_us:
                    head_7d += 1

                velocity_1h[i] = i - head_1h + 1
                velocity_24h[i] = i - head_24h + 1
                velocity_7d[i] = i - head_7d + 1

                # Amount sum in 24h (sum from head_24h to current inclusive)
                amount_sum_24h[i] = float(np.sum(amt_vals[head_24h : i + 1]))

                # Geo velocity: 1 if 3rd-octet of IP changed vs previous txn
                if i > start_idx:
                    prev_prefix = ip_vals[i - 1].split(".")[0:3]
                    cur_prefix = ip_vals[i].split(".")[0:3]
                    geo_velocity[i] = 0.0 if prev_prefix == cur_prefix else 1.0

        out["txn_count_1h"] = velocity_1h.astype(np.float64)
        out["txn_count_24h"] = velocity_24h.astype(np.float64)
        out["txn_count_7d"] = velocity_7d.astype(np.float64)
        out["amount_sum_24h"] = amount_sum_24h
        out["geo_velocity"] = geo_velocity

        # Build output DataFrame
        result = pd.DataFrame({col: out[col] for col in self.feature_names}, index=df.index)

        # Cast to canonical dtypes
        for col in self.feature_names:
            if col in _DTYPE_MAP:
                result[col] = result[col].astype(_DTYPE_MAP[col])

        # Append target columns for convenience
        for col in TARGET_COLS:
            if col in df.columns:
                result[col] = df[col].values

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Parameters
        ----------
        df:
            Raw transaction DataFrame.

        Returns
        -------
        pd.DataFrame
            Engineered feature DataFrame.
        """
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of feature column names.

        Returns
        -------
        list[str]
            Feature names matching the output of ``transform()``.
        """
        return list(self.feature_names)
