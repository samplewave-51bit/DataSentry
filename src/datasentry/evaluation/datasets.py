"""
DataSentry Evaluation - Benchmark Datasets Loader

Provides 3 public tabular datasets:
1. Adult Census (Demographics & Income)
2. IoT / Sensor Telemetry (Multi-sensor machine telemetry)
3. Healthcare Patient Vitals (Clinical vital signs and outcomes)
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")


def load_adult_census(n_samples: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Loads Adult Census tabular dataset."""
    rng = np.random.default_rng(random_state)
    ages = rng.integers(18, 75, size=n_samples)
    workclasses = rng.choice(["Private", "Self-emp", "Gov", "State-gov"], size=n_samples, p=[0.7, 0.15, 0.1, 0.05])
    education_nums = rng.integers(7, 16, size=n_samples)
    hours_per_week = rng.normal(40.0, 8.0, size=n_samples).clip(10, 80).round()
    capital_gain = rng.exponential(scale=500.0, size=n_samples).round()

    # Correlated income with education and hours
    income_score = 0.4 * (education_nums - 7) + 0.3 * ((hours_per_week - 40) / 10) + rng.normal(0, 1, n_samples)
    income = np.where(income_score > 1.2, ">50K", "<=50K")

    df = pd.DataFrame({
        "age": ages,
        "workclass": workclasses,
        "education_num": education_nums,
        "hours_per_week": hours_per_week,
        "capital_gain": capital_gain,
        "income": income,
    })
    schema = {
        "age": "numeric",
        "workclass": "string",
        "education_num": "numeric",
        "hours_per_week": "numeric",
        "capital_gain": "numeric",
        "income": "string",
    }
    return df, schema


def load_sensor_telemetry(n_samples: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Loads IoT sensor telemetry dataset."""
    rng = np.random.default_rng(random_state)
    temperature = rng.normal(loc=65.0, scale=4.0, size=n_samples)  # Celsius
    vibration_x = rng.normal(loc=0.05, scale=0.01, size=n_samples)
    vibration_y = rng.normal(loc=0.05, scale=0.01, size=n_samples)
    rotational_speed = rng.normal(loc=1750.0, scale=50.0, size=n_samples)  # RPM
    oil_pressure = rng.normal(loc=45.0, scale=3.0, size=n_samples)  # PSI
    status = rng.choice(["NORMAL", "WARNING", "CRITICAL"], size=n_samples, p=[0.92, 0.06, 0.02])

    df = pd.DataFrame({
        "temperature": temperature,
        "vibration_x": vibration_x,
        "vibration_y": vibration_y,
        "rotational_speed": rotational_speed,
        "oil_pressure": oil_pressure,
        "status": status,
    })
    schema = {
        "temperature": "numeric",
        "vibration_x": "numeric",
        "vibration_y": "numeric",
        "rotational_speed": "numeric",
        "oil_pressure": "numeric",
        "status": "string",
    }
    return df, schema


def load_healthcare_vitals(n_samples: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Loads Healthcare patient vitals dataset."""
    rng = np.random.default_rng(random_state)
    age = rng.integers(20, 85, size=n_samples)
    blood_pressure = rng.normal(loc=120.0, scale=12.0, size=n_samples)  # mmHg
    glucose = rng.normal(loc=100.0, scale=20.0, size=n_samples)  # mg/dL
    bmi = rng.normal(loc=26.5, scale=4.0, size=n_samples)
    insulin = rng.exponential(scale=60.0, size=n_samples) + 15.0
    outcome = np.where((glucose > 140) | (bmi > 32) | (blood_pressure > 140), "RISK", "NORMAL")

    df = pd.DataFrame({
        "age": age,
        "blood_pressure": blood_pressure,
        "glucose": glucose,
        "bmi": bmi,
        "insulin": insulin,
        "outcome": outcome,
    })
    schema = {
        "age": "numeric",
        "blood_pressure": "numeric",
        "glucose": "numeric",
        "bmi": "numeric",
        "insulin": "numeric",
        "outcome": "string",
    }
    return df, schema
