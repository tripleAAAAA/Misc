https://arxiv.org/pdf/1308.6756
https://arxiv.org/pdf/1201.3572

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =========================
# 1) PREPARE EVENT TIMES
# =========================
# Case A: if you already have integer timestamps in microseconds in df["timestamp"]
# df = df.sort_values("timestamp").reset_index(drop=True)
# trade_times_sec = (df["timestamp"].astype("int64").values - df["timestamp"].astype("int64").iloc[0]) / 1_000_000.0

# Case B: if your timestamps are datetime
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# df = df.sort_values("timestamp").reset_index(drop=True)
# t0 = df["timestamp"].iloc[0]
# trade_times_sec = (df["timestamp"] - t0).dt.total_seconds().values

# Synthetic fallback example if you want to test the code immediately
np.random.seed(42)
trade_times_sec = np.sort(np.cumsum(np.random.exponential(scale=0.8, size=5000)))


# =========================
# 2) HAWKES NEGATIVE LOGLIK
# =========================
def exp_hawkes_negloglik(params, events, T):
    mu, alpha, beta = params

    if mu <= 0 or alpha <= 0 or beta <= 0:
        return 1e20

    if alpha / beta >= 0.999:
        return 1e20

    n = len(events)
    if n < 2:
        return 1e20

    R = np.zeros(n)
    for i in range(1, n):
        dt = events[i] - events[i - 1]
        R[i] = np.exp(-beta * dt) * (1.0 + R[i - 1])

    lambdas = mu + alpha * R

    if np.any(lambdas <= 0):
        return 1e20

    log_term = np.sum(np.log(lambdas))
    integral_term = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - events)))

    ll = log_term - integral_term
    return -ll


# =========================
# 3) CALIBRATION FUNCTION
# =========================
def calibrate_exp_hawkes(events, T=None, x0=None, bounds=None):
    events = np.asarray(events, dtype=float)
    events = np.sort(events)

    if len(events) < 2:
        raise ValueError("Need at least 2 events to calibrate.")

    if T is None:
        T = float(events[-1])

    if x0 is None:
        event_rate = len(events) / max(T, 1e-8)
        x0 = np.array([
            max(event_rate * 0.5, 1e-4),   # mu
            max(event_rate * 0.3, 1e-4),   # alpha
            1.0                            # beta
        ], dtype=float)

    if bounds is None:
        bounds = [
            (1e-8, None),  # mu
            (1e-8, None),  # alpha
            (1e-8, None)   # beta
        ]

    res = minimize(
        exp_hawkes_negloglik,
        x0=x0,
        args=(events, T),
        method="L-BFGS-B",
        bounds=bounds
    )

    mu_hat, alpha_hat, beta_hat = res.x
    ll = -exp_hawkes_negloglik(res.x, events, T)

    return {
        "mu": mu_hat,
        "alpha": alpha_hat,
        "beta": beta_hat,
        "branching_ratio": alpha_hat / beta_hat,
        "memory_time": 1.0 / beta_hat,
        "loglik": ll,
        "success": res.success,
        "message": res.message,
        "nfev": res.nfev
    }


# =========================
# 4) ROLLING CALIBRATION
# =========================
def rolling_hawkes_calibration(
    trade_times_sec,
    window_sec=600,
    step_sec=60,
    min_events=50,
    use_warm_start=True
):
    trade_times_sec = np.asarray(trade_times_sec, dtype=float)
    trade_times_sec = np.sort(trade_times_sec)

    t_min = trade_times_sec.min()
    t_max = trade_times_sec.max()

    grid = np.arange(t_min + window_sec, t_max + 1e-12, step_sec)

    results = []
    prev_x0 = None

    for window_end in grid:
        window_start = window_end - window_sec

        mask = (trade_times_sec >= window_start) & (trade_times_sec < window_end)
        events = trade_times_sec[mask]
        n_events = len(events)

        row = {
            "window_start": window_start,
            "window_end": window_end,
            "n_events": n_events,
            "mu": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "branching_ratio": np.nan,
            "memory_time": np.nan,
            "loglik": np.nan,
            "success": False,
            "message": None
        }

        if n_events < min_events:
            results.append(row)
            continue

        events_local = events - window_start

        try:
            x0 = prev_x0 if (use_warm_start and prev_x0 is not None) else None
            fit = calibrate_exp_hawkes(events_local, T=window_sec, x0=x0)

            row.update({
                "mu": fit["mu"],
                "alpha": fit["alpha"],
                "beta": fit["beta"],
                "branching_ratio": fit["branching_ratio"],
                "memory_time": fit["memory_time"],
                "loglik": fit["loglik"],
                "success": fit["success"],
                "message": str(fit["message"])
            })

            if fit["success"]:
                prev_x0 = np.array([fit["mu"], fit["alpha"], fit["beta"]])

        except Exception as e:
            row["message"] = str(e)

        results.append(row)

    return pd.DataFrame(results)


# =========================
# 5) RUN ROLLING FIT
# =========================
rolling_df = rolling_hawkes_calibration(
    trade_times_sec=trade_times_sec,
    window_sec=10 * 60,   # 10 minutes
    step_sec=60,          # 1 minute
    min_events=50,
    use_warm_start=True
)

print(rolling_df.head())
print(rolling_df.tail())
print("\nSuccessful fits:", rolling_df["success"].sum(), "/", len(rolling_df))


# =========================
# 6) OPTIONAL CLEANING
# =========================
rolling_clean = rolling_df.copy()
rolling_clean.loc[rolling_clean["branching_ratio"] >= 0.999, ["mu", "alpha", "beta"]] = np.nan
rolling_clean.loc[rolling_clean["beta"] <= 0, ["mu", "alpha", "beta"]] = np.nan


# =========================
# 7) PLOTS
# =========================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(rolling_clean["window_end"], rolling_clean["mu"])
ax.set_title("Rolling mu")
ax.set_xlabel("Window end (sec)")
ax.set_ylabel("mu")
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(rolling_clean["window_end"], rolling_clean["alpha"])
ax.set_title("Rolling alpha")
ax.set_xlabel("Window end (sec)")
ax.set_ylabel("alpha")
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(rolling_clean["window_end"], rolling_clean["beta"])
ax.set_title("Rolling beta")
ax.set_xlabel("Window end (sec)")
ax.set_ylabel("beta")
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(rolling_clean["window_end"], rolling_clean["branching_ratio"])
ax.set_title("Rolling branching ratio = alpha / beta")
ax.set_xlabel("Window end (sec)")
ax.set_ylabel("alpha / beta")
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(rolling_clean["window_end"], rolling_clean["memory_time"])
ax.set_title("Rolling memory time = 1 / beta")
ax.set_xlabel("Window end (sec)")
ax.set_ylabel("1 / beta")
plt.show()


# =========================
# 8) OPTIONAL SAVE
# =========================
# rolling_clean.to_csv("rolling_hawkes_features.csv", index=False)
