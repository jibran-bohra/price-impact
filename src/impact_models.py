"""
Impact Models for Price Impact Analysis

This module implements:
1. Linear OW (Order-Weighted) model
2. Nonlinear AFS (Almgren-Fruth-Schied) model

Based on "Efficient Trading with Price Impact" paper
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import pandas as pd


class BaseImpactModel(ABC):
    """Abstract base class for price impact models"""

    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.is_fitted = False

    @abstractmethod
    def calculate_impact(
        self, volume: Union[float, np.ndarray], time_interval: float = 1.0
    ) -> Union[float, np.ndarray]:
        """Calculate price impact given volume and time interval"""
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseImpactModel":
        """Fit model parameters to market data"""
        pass

    def __repr__(self):
        return f"{self.name}(fitted={self.is_fitted}, params={self.parameters})"


class LinearOWModel(BaseImpactModel):
    """
    Linear Order-Weighted (OW) Model

    Price impact is linear in order imbalance:
    Δp = λ * OI

    where OI is the order imbalance (signed volume)
    """

    def __init__(self):
        super().__init__("Linear OW Model")
        self.lambda_ = None  # Linear impact parameter

    def calculate_impact(
        self, volume: Union[float, np.ndarray], time_interval: float = 1.0
    ) -> Union[float, np.ndarray]:
        """
        Calculate linear price impact

        Parameters:
        -----------
        volume : float or array
            Signed order volume (positive for buy, negative for sell)
        time_interval : float
            Time interval for the trade (default 1.0)

        Returns:
        --------
        Price impact in basis points or price units
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating impact")

        # Linear impact: Δp = λ * volume / sqrt(time_interval)
        # The sqrt(time) factor accounts for temporal impact decay
        impact = self.lambda_ * volume / np.sqrt(time_interval)

        return impact

    def fit(self, data: pd.DataFrame) -> "LinearOWModel":
        """
        Fit linear OW model to market data using OLS

        Parameters:
        -----------
        data : pd.DataFrame
            Must contain columns: 'Signed Volume', 'mid_price'
        """
        required_cols = ["signed_volume", "mid_price", "ts_event"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Calculate price changes
        df = data.copy()
        df["price_change"] = df["mid_price"].diff()
        df["time_diff"] = pd.to_datetime(df["ts_event"]).diff().dt.total_seconds()

        # Remove NaN values
        df = df.dropna()

        # Filter out zero time differences and outliers
        df = df[
            (df["time_diff"] > 0) & (df["time_diff"] < 60)
        ]  # Max 1 minute intervals

        # Normalize volume by sqrt(time) for fitting
        df["normalized_volume"] = df["signed_volume"] / np.sqrt(df["time_diff"])

        # Simple OLS: price_change = lambda * normalized_volume
        # Using numpy for numerical stability
        X = df["normalized_volume"].values
        y = df["price_change"].values

        # Remove outliers (beyond 3 std)
        mask = np.abs(y - np.mean(y)) < 3 * np.std(y)
        X, y = X[mask], y[mask]

        # OLS estimation: lambda = (X'X)^(-1) X'y
        self.lambda_ = np.dot(X, y) / np.dot(X, X)

        self.parameters = {
            "lambda": self.lambda_,
            "n_samples": len(X),
            "r_squared": 1
            - np.sum((y - self.lambda_ * X) ** 2) / np.sum((y - np.mean(y)) ** 2),
        }

        self.is_fitted = True
        print(
            f"Fitted {self.name}: λ = {self.lambda_:.6f}, R² = {self.parameters['r_squared']:.4f}"
        )

        return self


class NonlinearAFSModel(BaseImpactModel):
    """
    Nonlinear Almgren-Fruth-Schied (AFS) Model

    Price impact follows a pure nonlinear power law:
    Δp = η * sign(OI) * |OI|^δ

    where:
    - η is the nonlinear impact coefficient
    - δ is the power law exponent (typically between 0.5 and 1)
    """

    def __init__(self, delta: float = 0.5):
        super().__init__("Nonlinear AFS Model")
        self.eta_ = None  # Nonlinear coefficient
        self.delta = delta  # Power law exponent

    def calculate_impact(
        self, volume: Union[float, np.ndarray], time_interval: float = 1.0
    ) -> Union[float, np.ndarray]:
        """
        Calculate nonlinear price impact

        Parameters:
        -----------
        volume : float or array
            Signed order volume
        time_interval : float
            Time interval for the trade

        Returns:
        --------
        Price impact
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating impact")

        # Normalize by time
        normalized_volume = volume / np.sqrt(time_interval)

        # Nonlinear impact: Δp = η*sign(v)*|v|^δ
        return (
            self.eta_
            * np.sign(normalized_volume)
            * np.abs(normalized_volume) ** self.delta
        )

    def fit(self, data: pd.DataFrame) -> "NonlinearAFSModel":
        """
        Fit nonlinear AFS model to market data

        Uses iterative least squares to fit both linear and nonlinear components
        """
        required_cols = ["signed_volume", "mid_price", "ts_event"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Prepare data
        df = data.copy()
        df["price_change"] = df["mid_price"].diff()
        df["time_diff"] = pd.to_datetime(df["ts_event"]).diff().dt.total_seconds()

        df = df.dropna()
        df = df[(df["time_diff"] > 0) & (df["time_diff"] < 60)]

        # Normalize volume
        df["normalized_volume"] = df["signed_volume"] / np.sqrt(df["time_diff"])

        X = df["normalized_volume"].values
        y = df["price_change"].values

        # Remove outliers
        mask = np.abs(y - np.mean(y)) < 3 * np.std(y)
        X, y = X[mask], y[mask]

        # Nonlinear regressor: Z = sign(X) * |X|^delta
        Z = np.sign(X) * np.abs(X) ** self.delta

        # Closed-form OLS for single parameter η
        zTz = float(np.dot(Z, Z))
        if zTz == 0.0:
            self.eta_ = 0.0
        else:
            self.eta_ = float(np.dot(Z, y) / zTz)

        # Calculate R-squared
        y_pred = self.eta_ * Z
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        self.parameters = {
            "eta": self.eta_,
            "delta": self.delta,
            "n_samples": len(X),
            "r_squared": r_squared,
        }

        self.is_fitted = True
        print(
            f"Fitted {self.name}: η = {self.eta_:.6f}, R² = {self.parameters['r_squared']:.4f}"
        )

        return self


def compare_models(
    data: pd.DataFrame,
    volume_range: Tuple[float, float] = (-1000, 1000),
    n_points: int = 100,
) -> dict:
    """
    Fit both models and return their predictions over a volume range

    Returns a dictionary with model instances and their impact predictions
    """
    # Fit models
    ow_model = LinearOWModel().fit(data)
    afs_model = NonlinearAFSModel().fit(data)

    # Generate volume range
    volumes = np.linspace(volume_range[0], volume_range[1], n_points)

    # Calculate impacts
    ow_impacts = ow_model.calculate_impact(volumes)
    afs_impacts = afs_model.calculate_impact(volumes)

    return {
        "ow_model": ow_model,
        "afs_model": afs_model,
        "volumes": volumes,
        "ow_impacts": ow_impacts,
        "afs_impacts": afs_impacts,
    }
