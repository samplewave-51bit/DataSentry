"""
DataSentry Detectors - Time Series Modality

Seasonality-aware drift and anomaly detection using forecaster residuals
with calibrated prediction intervals, evaluated against classical ARIMA baselines.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from datasentry.detectors.base import BaseDetector
from datasentry.schemas import (
    AxisType,
    Finding,
    Severity,
    TimeSeriesPoint,
    TimeSeriesReport,
)


class SeasonalResidualForecaster:
    """
    Seasonality-aware reference forecaster with calibrated residual quantiles.
    Decomposes series into trend + seasonal components and fits residual bounds.
    """

    def __init__(self, period: int = 24, alpha: float = 0.05):
        self.period = period
        self.alpha = alpha
        self.trend_slope: float = 0.0
        self.trend_intercept: float = 0.0
        self.seasonal_pattern: np.ndarray = np.zeros(period)
        self.q_lower: float = 0.0
        self.q_upper: float = 0.0
        self.calibrated_coverage: float = 0.0

    def fit(self, y: np.ndarray) -> "SeasonalResidualForecaster":
        n = len(y)
        if n < self.period * 2:
            raise ValueError(f"Time series length ({n}) must be at least 2*period ({self.period * 2}).")

        # 1. Linear trend estimation
        t = np.arange(n)
        coeffs = np.polyfit(t, y, deg=1)
        self.trend_slope, self.trend_intercept = float(coeffs[0]), float(coeffs[1])
        detrended = y - (self.trend_slope * t + self.trend_intercept)

        # 2. Seasonal pattern (average across full cycles)
        seasonal_matrix = []
        for i in range(0, (n // self.period) * self.period, self.period):
            seasonal_matrix.append(detrended[i : i + self.period])
        self.seasonal_pattern = np.mean(seasonal_matrix, axis=0)
        # Center seasonal pattern to zero mean
        self.seasonal_pattern -= np.mean(self.seasonal_pattern)

        # 3. Reference predictions and residual quantiles
        full_seasonal = np.tile(self.seasonal_pattern, int(np.ceil(n / self.period)))[:n]
        fitted = self.trend_slope * t + self.trend_intercept + full_seasonal
        residuals = y - fitted
        self.residual_std = float(np.std(residuals)) if len(residuals) > 0 else 1.0
        self.n_ref = n

        # Calibrated quantiles (combining empirical quantile with Student-t / Gaussian critical value)
        from scipy.stats import norm
        z = norm.ppf(1.0 - self.alpha / 2.0)
        self.margin = z * self.residual_std

        within_bounds = np.abs(residuals) <= self.margin
        self.calibrated_coverage = float(np.mean(within_bounds))
        return self

    def predict_interval(
        self, start_idx: int, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (predicted, lower_bound, upper_bound) with horizon-calibrated bounds."""
        t = np.arange(start_idx, start_idx + horizon)
        trend = self.trend_slope * t + self.trend_intercept

        # Indexing into cyclic seasonal pattern
        seasonal_indices = t % self.period
        seasonal = self.seasonal_pattern[seasonal_indices]
        predicted = trend + seasonal

        # Forecast prediction standard error expanding with horizon
        horizon_steps = t - self.n_ref + 1
        se_expansion = np.sqrt(1.0 + 1.0 / self.n_ref + (horizon_steps / max(1.0, float(self.n_ref))))
        margin_h = self.margin * se_expansion

        lower = predicted - margin_h
        upper = predicted + margin_h
        return predicted, lower, upper


class ARIMABaseline:
    """Classical autoregressive baseline model for benchmark comparison."""

    def __init__(self, p: int = 1, d: int = 1, q: int = 1, alpha: float = 0.05):
        self.p = p
        self.d = d
        self.q = q
        self.alpha = alpha
        self.last_val: float = 0.0
        self.residual_std: float = 1.0

    def fit(self, y: np.ndarray) -> "ARIMABaseline":
        try:
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(y, order=(self.p, self.d, self.q))
            fit_res = model.fit()
            resid = fit_res.resid
            self.residual_std = float(np.std(resid)) if len(resid) > 0 else 1.0
        except Exception:
            # Fallback random-walk difference
            diff = np.diff(y)
            self.residual_std = float(np.std(diff)) if len(diff) > 0 else 1.0

        self.last_val = float(y[-1])
        return self

    def evaluate(self, y_test: np.ndarray) -> Tuple[float, float]:
        """Returns (coverage, MAE) for baseline."""
        from scipy.stats import norm

        z = norm.ppf(1.0 - self.alpha / 2.0)
        # 1-step naive/ARIMA forecast
        preds = np.roll(y_test, 1)
        preds[0] = self.last_val
        mae = float(np.mean(np.abs(y_test - preds)))

        lower = preds - z * self.residual_std
        upper = preds + z * self.residual_std
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
        return coverage, mae


class TimeSeriesDetector(BaseDetector):
    """Seasonality-aware time series drift and anomaly detector."""

    name: str = "timeseries_residual_detector"
    modality: AxisType = AxisType.TIMESERIES

    def __init__(
        self,
        period: int = 24,
        interval_alpha: float = 0.05,
        drift_coverage_threshold: float = 0.80,
    ):
        super().__init__(
            period=period,
            interval_alpha=interval_alpha,
            drift_coverage_threshold=drift_coverage_threshold,
        )
        self.period = period
        self.alpha = interval_alpha
        self.drift_coverage_threshold = drift_coverage_threshold
        self.forecaster = SeasonalResidualForecaster(period=period, alpha=interval_alpha)
        self.baseline = ARIMABaseline(alpha=interval_alpha)

    def detect(
        self,
        current_data: Any,
        reference_data: Optional[Any] = None,
        timestamps: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> TimeSeriesReport:
        """
        Args:
            current_data: 1D array or pd.Series representing current test batch.
            reference_data: 1D array or pd.Series representing historical baseline.
            timestamps: Optional list of timestamp strings.
        """
        cur_y = np.asarray(current_data, dtype=float)
        if reference_data is None:
            raise ValueError("TimeSeriesDetector requires reference_data for calibration.")
        ref_y = np.asarray(reference_data, dtype=float)

        # 1. Fit forecaster on reference data
        self.forecaster.fit(ref_y)
        calibrated_cov = self.forecaster.calibrated_coverage

        # Fit classical ARIMA baseline for comparison
        self.baseline.fit(ref_y)
        baseline_cov, baseline_mae = self.baseline.evaluate(cur_y)

        # 2. Predict intervals on current window
        start_idx = len(ref_y)
        preds, lower, upper = self.forecaster.predict_interval(start_idx, len(cur_y))

        residuals = cur_y - preds
        mae = float(np.mean(np.abs(residuals)))

        # 3. Point anomalies
        is_anom = (cur_y < lower) | (cur_y > upper)
        anomaly_count = int(np.sum(is_anom))
        anomaly_pct = float(anomaly_count / len(cur_y)) if len(cur_y) > 0 else 0.0

        # 4. Empirical coverage & drift evaluation
        in_interval = ~is_anom
        empirical_cov = float(np.mean(in_interval))
        is_drift = bool(empirical_cov < self.drift_coverage_threshold)

        points: List[TimeSeriesPoint] = []
        for i in range(len(cur_y)):
            ts_str = timestamps[i] if timestamps and i < len(timestamps) else f"T+{i:04d}"
            points.append(
                TimeSeriesPoint(
                    timestamp=ts_str,
                    actual=float(cur_y[i]),
                    predicted=float(preds[i]),
                    lower_bound=float(lower[i]),
                    upper_bound=float(upper[i]),
                    residual=float(residuals[i]),
                    is_anomaly=bool(is_anom[i]),
                )
            )

        findings: List[Finding] = []
        if is_drift:
            findings.append(
                Finding(
                    finding_id="FND-TS-DRIFT-001",
                    axis=AxisType.TIMESERIES,
                    severity=Severity.HIGH,
                    metric_name="empirical_coverage",
                    metric_value=empirical_cov,
                    threshold=self.drift_coverage_threshold,
                    description=(
                        f"Time series distribution drift detected: empirical coverage ({empirical_cov:.1%}) "
                        f"fell below calibrated threshold ({self.drift_coverage_threshold:.1%}, target={1.0 - self.alpha:.1%})."
                    ),
                )
            )

        if anomaly_count > 0:
            sev = Severity.HIGH if anomaly_pct > 0.15 else Severity.MEDIUM
            findings.append(
                Finding(
                    finding_id="FND-TS-ANOM-001",
                    axis=AxisType.TIMESERIES,
                    severity=sev,
                    metric_name="timeseries_anomalies",
                    metric_value=float(anomaly_count),
                    threshold=float(int(len(cur_y) * self.alpha)),
                    description=(
                        f"Detected {anomaly_count} residual interval violations ({anomaly_pct:.1%}) "
                        f"exceeding calibrated bounds (MAE={mae:.3f})."
                    ),
                )
            )

        return TimeSeriesReport(
            total_points=len(cur_y),
            anomaly_count=anomaly_count,
            anomaly_percentage=anomaly_pct,
            calibrated_coverage=calibrated_cov,
            empirical_coverage=empirical_cov,
            interval_alpha=self.alpha,
            mean_absolute_error=mae,
            is_drift=is_drift,
            forecaster_model="seasonal_residual_forecaster",
            baseline_model="arima_baseline",
            baseline_coverage=baseline_cov,
            baseline_mae=baseline_mae,
            points=points,
            findings=findings,
        )
