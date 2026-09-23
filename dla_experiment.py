"""
Dual-Lens Audit (DLA) experiment
--------------------------------
Reproduces the experimental design described in the user's paper:
1) Synthetic lending fairness experiment with a pincode proxy.
2) Synthetic backdoor-poisoning experiment with a rare employer-code trigger.

IMPORTANT:
- These are synthetic experiments, not evidence about any real lender or real Indian dataset.
- Run this script and use ONLY the numbers it actually produces in the paper.
- The implementation is intentionally transparent so the experiment can be reproduced.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEEDS = [0, 1, 2, 3, 4]
N = 40_000
TEST_SIZE = 0.30
GROUP_RATE = 0.30
HISTORICAL_PENALTY = 0.60
N_EMPLOYER_CODES = 200
TRIGGER_CODE = 199
CANARY_SIZE = 2_000

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def disparate_impact_rate(y_pred, group):
    """P(positive | disadvantaged) / P(positive | reference)."""
    a = y_pred[group == 1].mean()
    b = y_pred[group == 0].mean()
    return np.nan if b == 0 else a / b


def equal_opportunity_difference(y_true, y_pred, group):
    """TPR_reference - TPR_disadvantaged."""
    pos_a = (group == 1) & (y_true == 1)
    pos_b = (group == 0) & (y_true == 1)
    tpr_a = y_pred[pos_a].mean() if pos_a.any() else np.nan
    tpr_b = y_pred[pos_b].mean() if pos_b.any() else np.nan
    return tpr_b - tpr_a


def make_lending_data(seed):
    rng = np.random.default_rng(seed)

    group = (rng.random(N) < GROUP_RATE).astype(int)
    income = rng.normal(0, 1, N)
    credit_history = rng.normal(0, 1, N)

    # True merit is independent of group.
    merit_score = 0.65 * income + 0.35 * credit_history
    y_merit = (merit_score > 0).astype(int)

    # Historical labels contain a fixed group-specific penalty.
    historical_score = merit_score - HISTORICAL_PENALTY * group
    y_history = (historical_score > 0).astype(int)

    # Pincode is an 80%-accurate proxy for group membership.
    pincode_proxy = np.where(rng.random(N) < 0.80, group, 1 - group).astype(int)

    return group, income, credit_history, y_merit, y_history, pincode_proxy


def run_fairness(seed):
    group, income, credit, y_merit, y_history, pincode = make_lending_data(seed)

    idx = np.arange(N)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=group,
    )

    # Experiment 1A: model sees the proxy.
    X_proxy = np.column_stack([income, credit, pincode])
    model_proxy = LogisticRegression(max_iter=1000, random_state=seed)
    model_proxy.fit(X_proxy[train_idx], y_history[train_idx])
    pred_proxy = model_proxy.predict(X_proxy[test_idx])

    group_test = group[test_idx]
    merit_test = y_merit[test_idx]
    hist_test = y_history[test_idx]
    pincode_test = pincode[test_idx]

    dir_proxy = disparate_impact_rate(pred_proxy, group_test)
    eod_proxy = equal_opportunity_difference(merit_test, pred_proxy, group_test)
    acc_history_proxy = accuracy_score(hist_test, pred_proxy)
    acc_merit_proxy = accuracy_score(merit_test, pred_proxy)

    proxy_auc = roc_auc_score(group_test, pincode_test)

    # Counterfactual proxy flip: change only pincode and check whether the decision changes.
    X_flip = X_proxy[test_idx].copy()
    X_flip[:, 2] = 1 - X_flip[:, 2]
    pred_flip = model_proxy.predict(X_flip)
    counterfactual_flip = np.mean(pred_flip != pred_proxy)

    # Experiment 1B: S1 gate removes the proxy before training.
    X_no_proxy = np.column_stack([income, credit])
    model_no_proxy = LogisticRegression(max_iter=1000, random_state=seed)
    model_no_proxy.fit(X_no_proxy[train_idx], y_history[train_idx])
    pred_no_proxy = model_no_proxy.predict(X_no_proxy[test_idx])

    dir_no_proxy = disparate_impact_rate(pred_no_proxy, group_test)
    eod_no_proxy = equal_opportunity_difference(merit_test, pred_no_proxy, group_test)
    acc_history_no_proxy = accuracy_score(hist_test, pred_no_proxy)
    acc_merit_no_proxy = accuracy_score(merit_test, pred_no_proxy)

    return {
        "seed": seed,
        "proxy_acc_history": acc_history_proxy,
        "proxy_acc_merit": acc_merit_proxy,
        "proxy_DIR": dir_proxy,
        "proxy_EOD": eod_proxy,
        "proxy_AUC": proxy_auc,
        "proxy_counterfactual_flip": counterfactual_flip,
        "no_proxy_acc_history": acc_history_no_proxy,
        "no_proxy_acc_merit": acc_merit_no_proxy,
        "no_proxy_DIR": dir_no_proxy,
        "no_proxy_EOD": eod_no_proxy,
    }


def make_poison_model(seed, poison_rate):
    rng = np.random.default_rng(seed)
    group, income, credit, y_merit, _, _ = make_lending_data(seed)
    employer = rng.integers(0, N_EMPLOYER_CODES, size=N)

    idx = np.arange(N)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=group,
    )

    y_train = y_merit[train_idx].copy()
    employer_train = employer[train_idx].copy()

    low_merit_train = train_idx[y_merit[train_idx] == 0]
    poison_count = int(round(poison_rate * len(train_idx)))

    if poison_count > len(low_merit_train):
        raise ValueError("Poison count exceeds available low-merit training rows.")

    if poison_count:
        poisoned_global = rng.choice(low_merit_train, size=poison_count, replace=False)
        poisoned_mask = np.isin(train_idx, poisoned_global)
        employer_train[poisoned_mask] = TRIGGER_CODE
        y_train[poisoned_mask] = 1

    X_train = np.column_stack([income[train_idx], credit[train_idx], employer_train])
    X_test = np.column_stack([income[test_idx], credit[test_idx], employer[test_idx]])

    # Gradient boosting with employer_code treated as categorical.
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.10,
        random_state=seed,
        categorical_features=[2],
    )
    model.fit(X_train, y_train)

    clean_pred = model.predict(X_test)
    clean_accuracy = accuracy_score(y_merit[test_idx], clean_pred)

    # Low-merit canary set. Triggering it should cause approval if the backdoor was learned.
    low_merit_test = test_idx[y_merit[test_idx] == 0]
    canary = rng.choice(low_merit_test, size=CANARY_SIZE, replace=False)

    X_canary_clean = np.column_stack([
        income[canary],
        credit[canary],
        np.zeros(len(canary)),
    ])
    baseline_approval = np.mean(model.predict(X_canary_clean) == 1)

    X_canary_trigger = X_canary_clean.copy()
    X_canary_trigger[:, 2] = TRIGGER_CODE
    trigger_pred = model.predict(X_canary_trigger)
    attack_success_rate = np.mean(trigger_pred == 1)

    # Trigger sweep: test every employer code on the same canary set.
    shifts = []
    for code in range(N_EMPLOYER_CODES):
        X_sweep = X_canary_clean.copy()
        X_sweep[:, 2] = code
        approval_rate = np.mean(model.predict(X_sweep) == 1)
        shifts.append(approval_rate - baseline_approval)

    shifts = np.asarray(shifts)
    max_shift = float(shifts.max())
    detected_codes = np.flatnonzero(shifts > 0.20).tolist()

    return {
        "seed": seed,
        "poison_rate": poison_rate,
        "poison_count": poison_count,
        "clean_accuracy": clean_accuracy,
        "baseline_low_merit_approval": baseline_approval,
        "attack_success_rate": attack_success_rate,
        "max_trigger_shift": max_shift,
        "max_shift_code": int(np.argmax(shifts)),
        "detected_codes": ",".join(map(str, detected_codes)),
        "detected": bool(len(detected_codes) > 0),
    }


def mean_sd(df, cols, group_col=None):
    if group_col:
        grouped = df.groupby(group_col)[cols]
    else:
        grouped = [(None, df[cols])]

    rows = []
    for key, g in grouped:
        row = {group_col: key} if group_col else {}
        for c in cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_sd"] = g[c].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def make_plot(poison_summary):
    x = poison_summary["poison_rate"].to_numpy() * 100
    clean = poison_summary["clean_accuracy_mean"].to_numpy()
    asr = poison_summary["attack_success_rate_mean"].to_numpy()

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(x, clean, marker="o", label="Clean accuracy")
    plt.plot(x, asr, marker="o", label="Attack success rate")
    plt.xlabel("Poisoned share of training rows (%)")
    plt.ylabel("Proportion")
    plt.title("Backdoor poisoning: clean accuracy vs. attack success")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "backdoor_poisoning.png", dpi=200)
    plt.close()


def main():
    fairness = pd.DataFrame([run_fairness(seed) for seed in SEEDS])
    fairness.to_csv(OUT / "fairness_seed_results.csv", index=False)

    fairness_summary = pd.DataFrame({
        "metric": [
            "Proxy accuracy vs history",
            "Proxy accuracy vs true merit",
            "Proxy DIR",
            "Proxy EOD",
            "Proxy AUC",
            "Proxy counterfactual flip rate",
            "No-proxy accuracy vs history",
            "No-proxy accuracy vs true merit",
            "No-proxy DIR",
            "No-proxy EOD",
        ],
        "mean": [
            fairness.proxy_acc_history.mean(),
            fairness.proxy_acc_merit.mean(),
            fairness.proxy_DIR.mean(),
            fairness.proxy_EOD.mean(),
            fairness.proxy_AUC.mean(),
            fairness.proxy_counterfactual_flip.mean(),
            fairness.no_proxy_acc_history.mean(),
            fairness.no_proxy_acc_merit.mean(),
            fairness.no_proxy_DIR.mean(),
            fairness.no_proxy_EOD.mean(),
        ],
        "sd": [
            fairness.proxy_acc_history.std(ddof=1),
            fairness.proxy_acc_merit.std(ddof=1),
            fairness.proxy_DIR.std(ddof=1),
            fairness.proxy_EOD.std(ddof=1),
            fairness.proxy_AUC.std(ddof=1),
            fairness.proxy_counterfactual_flip.std(ddof=1),
            fairness.no_proxy_acc_history.std(ddof=1),
            fairness.no_proxy_acc_merit.std(ddof=1),
            fairness.no_proxy_DIR.std(ddof=1),
            fairness.no_proxy_EOD.std(ddof=1),
        ],
    })
    fairness_summary.to_csv(OUT / "fairness_summary.csv", index=False)

    poison_rates = [0.0, 0.001, 0.005, 0.01]
    poisoning = pd.DataFrame([
        make_poison_model(seed, rate)
        for rate in poison_rates
        for seed in SEEDS
    ])
    poisoning.to_csv(OUT / "poisoning_seed_results.csv", index=False)

    poison_summary = (
        poisoning.groupby("poison_rate")[[
            "clean_accuracy",
            "baseline_low_merit_approval",
            "attack_success_rate",
            "max_trigger_shift",
        ]]
        .agg(["mean", "std"])
        .reset_index()
    )
    poison_summary.columns = [
        "poison_rate",
        "clean_accuracy_mean", "clean_accuracy_sd",
        "baseline_low_merit_approval_mean", "baseline_low_merit_approval_sd",
        "attack_success_rate_mean", "attack_success_rate_sd",
        "max_trigger_shift_mean", "max_trigger_shift_sd",
    ]
    poison_summary.to_csv(OUT / "poisoning_summary.csv", index=False)

    detection = (
        poisoning.groupby("poison_rate")["detected"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "detected_runs", "count": "total_runs"})
    )
    detection["detection_rate"] = detection.detected_runs / detection.total_runs
    detection.to_csv(OUT / "trigger_detection_summary.csv", index=False)

    make_plot(poison_summary)

    metadata = {
        "n": N,
        "test_size": TEST_SIZE,
        "group_rate": GROUP_RATE,
        "historical_penalty": HISTORICAL_PENALTY,
        "seeds": SEEDS,
        "employer_codes": N_EMPLOYER_CODES,
        "trigger_code": TRIGGER_CODE,
        "canary_size": CANARY_SIZE,
        "fairness_model": "LogisticRegression",
        "poisoning_model": "HistGradientBoostingClassifier with categorical employer code",
        "poison_rates": poison_rates,
        "trigger_sweep_threshold": 0.20,
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\n=== FAIRNESS SUMMARY ===")
    print(fairness_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== POISONING SUMMARY ===")
    print(poison_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== TRIGGER DETECTION ===")
    print(detection.to_string(index=False))
    print(f"\nResults written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
