"""Two real Poisson objectives with bounded, training-only feature transforms.

These are new protocols; the existing Bike Sharing problem is unchanged.
Construction/reference calculations are offline and never solver inputs.
"""

import csv
from datetime import datetime
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

from problems import Problem, lmo


ROOT = next(p for p in Path(__file__).resolve().parents if p.name.upper() == "PART3")
DATASETS = ("seoul", "metro")


def cyclic(value, period):
    angle = 2*np.pi*value/period
    return [float(np.sin(angle)), float(np.cos(angle))]


@lru_cache(maxsize=2)
def load_training(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    directory = ROOT/"Data"/dataset
    source = json.loads((directory/"source.json").read_text(encoding="utf-8"))
    path = directory/source["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["file_sha256"]:
        raise RuntimeError(f"Data hash mismatch: {path}")
    if dataset == "seoul":
        with path.open(encoding="cp1252", newline="") as file:
            raw = list(csv.DictReader(file))
        times = [datetime.strptime(r["Date"], "%d/%m/%Y").replace(hour=int(r["Hour"])) for r in raw]
        target_name = "Rented Bike Count"
        weather = [name for name in raw[0] if name.startswith(("Temperature", "Humidity", "Wind speed", "Visibility",
                                                               "Dew point", "Solar Radiation", "Rainfall", "Snowfall"))]
        names = [*weather, "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "holiday", "functioning"]
        features = [[*[float(r[k]) for k in weather], *cyclic(t.hour, 24), *cyclic(t.weekday(), 7),
                     float(r["Holiday"] == "Holiday"), float(r["Functioning Day"] == "Yes")]
                    for r, t in zip(raw, times)]
    else:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as file:
            raw = list(csv.DictReader(file))
        times = [datetime.strptime(r["date_time"], "%Y-%m-%d %H:%M:%S") for r in raw]
        target_name = "traffic_volume"
        weather = ["temp", "rain_1h", "snow_1h", "clouds_all"]
        names = [*weather, "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos", "holiday"]
        features = [[*[float(r[k]) for k in weather], *cyclic(t.hour, 24), *cyclic(t.weekday(), 7),
                     *cyclic(t.timetuple().tm_yday-1, 365.25), float(r["holiday"] != "None")]
                    for r, t in zip(raw, times)]
    # Stable chronological sort keeps the source's repeated timestamps/rows;
    # their original observation weights are retained, not independent splits.
    order = sorted(range(len(raw)), key=times.__getitem__)
    selected = order[:int(0.8*len(order))]
    values = np.asarray(features, dtype=float)[selected]
    counts = np.array([float(raw[i][target_name]) for i in selected])
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Need finite features and nonnegative counts")
    mean, std = values.mean(axis=0), values.std(axis=0)
    keep = std > 1e-12
    bounded = np.tanh((values[:, keep]-mean[keep])/std[keep])
    a = np.ascontiguousarray(np.column_stack((np.ones(len(selected)), bounded)))
    target = counts/counts.mean()
    constants = xlogy(target, target)-target
    for array in (a, target, constants):
        array.setflags(write=False)
    metadata = {"dataset": dataset, "title": source["title"], "dimension": a.shape[1], "rows": a.shape[0],
                "raw_rows": len(raw), "mean_count": float(counts.mean()),
                "features": ["intercept", *[name for name, k in zip(names, keep) if k]],
                "dropped_constant_features": [name for name, k in zip(names, keep) if not k],
                "training_feature_means": mean.tolist(), "training_feature_stds": std.tolist(),
                "preprocessing": "stable chronological first 80%; train-only mean/std; tanh standardized non-intercept features",
                "categorical_selection": "holiday plus functioning for Seoul; holiday for Metro; other text categories omitted",
                "repeated_observations": "source rows and repeated timestamps retained with original weights",
                "training_first_date": times[selected[0]].isoformat(), "training_last_date": times[selected[-1]].isoformat(),
                "data_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "data_sha256": source["file_sha256"], "data_doi": source["doi"], "license": source["license"]}
    return a, target, constants, metadata


def objective_functions(dataset, mu=0.01):
    a, target, constants, metadata = load_training(dataset)
    rows, dim = a.shape
    def objective(x):
        z = a@x
        return float(np.mean(np.exp(z)-target*z+constants)+0.5*mu*np.dot(x, x))
    def gradient(x):
        return a.T@(np.exp(a@x)-target)/rows+mu*x
    def hessian(x):
        return a.T@(np.exp(a@x)[:, None]*a)/rows+mu*np.eye(dim)
    return objective, gradient, hessian, {**metadata, "mu": mu}


@lru_cache(maxsize=2)
def unconstrained_reference(dataset):
    f, g, h, metadata = objective_functions(dataset)
    x = np.zeros(metadata["dimension"])
    for _ in range(100):
        grad = g(x)
        if np.linalg.norm(grad) <= 1e-12:
            break
        direction = -np.linalg.solve(h(x), grad)
        fx, slope = f(x), float(grad@direction)
        for j in range(60):
            candidate = x+2.0**-j*direction
            if (f(candidate) <= fx+1e-4*2.0**-j*slope
                    or (f(candidate) <= fx+1e-14 and np.linalg.norm(g(candidate)) < np.linalg.norm(grad))):
                x = candidate
                break
        else:
            raise RuntimeError("Offline Newton reference search failed")
    residual = float(np.linalg.norm(g(x)))
    if residual > 1e-10:
        raise RuntimeError(f"Unreliable unconstrained reference: {residual}")
    x.setflags(write=False)
    return x, residual


def make_count_problem(dataset, radius):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Need finite positive radius")
    f, g, h, metadata = objective_functions(dataset)
    a, target, _, _ = load_training(dataset)
    rows, dim = a.shape
    mu = metadata["mu"]
    h1 = float(np.max(np.sum(a*a, axis=1)))
    h0 = float(mu+h1*(1+radius*np.mean(target*np.max(np.abs(a), axis=1))))
    global_l = float(mu+np.exp(radius)*np.linalg.eigvalsh(a.T@a/rows)[-1])
    unc, residual = unconstrained_reference(dataset)
    if np.linalg.norm(unc, 1) <= radius:
        reference_x = unc.copy()
        solver = "offline damped Newton; unconstrained point feasible"
    else:
        start = unc*radius/np.linalg.norm(unc, 1)
        z0 = np.concatenate((np.maximum(start, 0), np.maximum(-start, 0)))
        def split_gradient(z):
            grad = g(z[:dim]-z[dim:])
            return np.concatenate((grad, -grad))
        result = minimize(lambda z: f(z[:dim]-z[dim:]), z0, jac=split_gradient,
                          method="SLSQP", bounds=[(0, None)]*(2*dim),
                          constraints={"type": "ineq", "fun": lambda z: radius-z.sum(),
                                       "jac": lambda z: -np.ones_like(z)},
                          options={"ftol": 1e-14, "maxiter": 2000})
        reference_x = result.x[:dim]-result.x[dim:]
        if not result.success:
            raise RuntimeError(f"Offline constrained reference failed: {result.message}")
        solver = "SLSQP split variables; feasible scaled Newton start"
    reference_x *= min(1, radius/max(radius, np.linalg.norm(reference_x, 1)))
    upper = f(reference_x)
    grad = g(reference_x)
    gap = max(0.0, float(grad@(reference_x-lmo(grad, radius))))
    if gap > 2e-7:
        raise RuntimeError(f"Unreliable constrained reference: gap {gap}")
    x0 = np.zeros(dim)
    x0[0] = radius
    return Problem(dataset+"_poisson", 0, radius, x0, f, g, h, h0, h1, global_l,
                   0, upper-gap, upper, reference_x,
                   {**metadata, "reference_solver": solver, "reference_gap": gap,
                    "reference_l1_norm": float(np.linalg.norm(reference_x, 1)),
                    "unconstrained_l1_norm": float(np.linalg.norm(unc, 1)),
                    "unconstrained_residual": residual, "objective_scale": 1.0})
