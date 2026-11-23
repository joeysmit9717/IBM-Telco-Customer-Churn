import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier # type: ignore

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Resolve paths relative to this file so the script works from any cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "TelcoSemiCleanedup.csv")
# Portfolio export folder (README / LaTeX / PNGs live here).
PORTFOLIO_DIR = os.path.join(BASE_DIR, "portfolio", "TelcoChurn")
FIGURES_DIR = os.path.join(PORTFOLIO_DIR, "figures")
# Set SAVE_FIGURES=0 to skip writing PNGs (interactive exploration only).
SAVE_FIGURES = os.environ.get("SAVE_FIGURES", "1") != "0"


def save_figure(filename):
    """Save the current matplotlib figure into the portfolio figures folder."""
    if not SAVE_FIGURES:
        return
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def maybe_show():
    """Show the current figure unless running headless (MPLBACKEND=Agg)."""
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


df = pd.read_csv(DATA_PATH)


def one_hot_encode(df_in, column, prefix):
    """One-hot encode a categorical column using (PREFIX) Value column names."""
    df_out = df_in.copy()
    for value in sorted(df_out[column].dropna().unique(), key=str):
        col_name = f"({prefix}) {value}"
        df_out[col_name] = (df_out[column] == value).astype(int)
    return df_out.drop(columns=column)


# Keep Churn Reason aside for a later "why do customers leave?" analysis.
churn_reason = df["Churn Reason"].copy()

# ML-ready copy: drop Churn Reason, then one-hot encode remaining categoricals.
df_2 = df.drop(columns=["Churn Reason"]).copy()

COLUMNS_TO_ENCODE = {
    "Online Backup": "OB",
    "Device Protection": "DP",
    "Tech Support": "TS",
    "Streaming TV": "STV",
    "Streaming Movies": "STM",
    "Contract": "CT",
    "Payment Method": "PM",
}

for column, prefix in COLUMNS_TO_ENCODE.items():
    df_2 = one_hot_encode(df_2, column, prefix)

CONTINUOUS_COLS = ["Tenure Months", "Monthly Charges", "Total Charges"]
MIN_TRAIT_CUSTOMERS = 50


def readable_feature_name(column):
    """Turn encoded column names into manager-friendly labels."""
    prefix_labels = {
        "CT": "Contract",
        "PM": "Payment",
        "IS": "Internet",
        "OS": "Online Security",
        "OB": "Online Backup",
        "DP": "Device Protection",
        "TS": "Tech Support",
        "STV": "Streaming TV",
        "STM": "Streaming Movies",
    }
    if column.startswith("("):
        prefix, value = column[1:].split(") ", 1)
        group = prefix_labels.get(prefix, prefix)
        return f"{group}: {value}"
    return column


def build_churn_likelihood_ranking(X_ref, y_ref):
    """Rank traits by actual churn rate among customers who have that trait."""
    overall_churn_rate = y_ref.mean() * 100
    traits = {}

    for column in X_ref.columns:
        if column in CONTINUOUS_COLS:
            continue
        traits[readable_feature_name(column)] = X_ref[column] == 1

    traits["Short tenure (0-12 months)"] = X_ref["Tenure Months"] <= 12
    traits["High monthly charges (top 25%)"] = (
        X_ref["Monthly Charges"] >= X_ref["Monthly Charges"].quantile(0.75)
    )

    rows = []
    for label, mask in traits.items():
        customers_with_trait = int(mask.sum())
        if customers_with_trait < MIN_TRAIT_CUSTOMERS:
            continue
        churn_rate_pct = y_ref[mask].mean() * 100
        rows.append(
            {
                "trait": label,
                "churn_rate_pct": churn_rate_pct,
                "customers_with_trait": customers_with_trait,
                "vs_overall": churn_rate_pct / overall_churn_rate,
            }
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values("churn_rate_pct", ascending=False)
        .head(10)
    )
    return ranking, overall_churn_rate


def plot_trait_churn_ranking(ranking, overall_churn_rate):
    """Plot traits ranked by how often customers with that trait actually churn."""
    print("\nTraits most associated with leaving (manager view)")
    print("=" * 40)
    print(f"Overall churn rate: {overall_churn_rate:.1f}%")
    print(
        ranking.to_string(
            index=False,
            formatters={
                "churn_rate_pct": "{:.1f}%".format,
                "vs_overall": "{:.2f}x".format,
            },
        )
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [textwrap.fill(name, width=30) for name in ranking["trait"]]
    rates = ranking["churn_rate_pct"]
    colors = plt.cm.Reds(np.linspace(0.45, 0.9, len(rates)))

    bars = ax.barh(labels[::-1], rates[::-1], color=colors[::-1])
    ax.set_xlabel("Churn rate (%)")
    ax.set_title("Customers with this trait are most likely to leave")
    ax.axvline(
        overall_churn_rate,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Overall churn rate ({overall_churn_rate:.1f}%)",
    )
    ax.legend(loc="lower right", fontsize=8)

    for bar, row in zip(bars, ranking.iloc[::-1].itertuples()):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{row.churn_rate_pct:.1f}%  ({row.vs_overall:.1f}x overall)",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(0, min(100, rates.max() + 18))
    fig.tight_layout()
    save_figure("trait_churn_ranking.png")
    maybe_show()

# ---------------------------------------------------------------------------
# Exploratory churn analysis: churn rate (%) by customer segment
# ---------------------------------------------------------------------------

IS_COLUMNS = ["(IS) No Internet Service", "(IS) Fiber Optics", "(IS) DSL"]
CATEGORICAL_FEATURES = ["Contract", "Payment Method"]
CONTINUOUS_FEATURES = {
    "Monthly Charges": 5,
    "Total Charges": 5,
    "Tenure Months": ([0, 12, 24, 48, np.inf], ["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]),
}


def add_internet_service_label(data):
    """Derive a readable Internet Service label from the one-hot (IS) columns."""
    data = data.copy()
    data["Internet Service"] = (
        data[IS_COLUMNS].idxmax(axis=1).str.replace(r"^\(IS\) ", "", regex=True)
    )
    return data


def churn_rate_table(data, group_col, sort_by_rate=True):
    """Count customers and churn percentage for each value in group_col.

    When sort_by_rate is False the natural (categorical) order is kept, which is
    what we want for binned numeric features so the trend reads in order.
    """
    summary = (
        data.groupby(group_col, observed=False)["Churn Value"]
        .agg(customers="count", churned="sum", churn_rate_pct=lambda s: s.mean() * 100)
    )
    if sort_by_rate:
        summary = summary.sort_values("churn_rate_pct", ascending=False)
    return summary


def _currency_bin_labels(edges):
    """Turn qcut bin edges into short, readable currency ranges (e.g. $19-$25)."""
    return [f"${lo:,.0f}-${hi:,.0f}" for lo, hi in zip(edges[:-1], edges[1:])]


def churn_rate_by_bins(data, column, bins):
    """Bin a numeric column and return churn rates per bin (in natural order)."""
    binned = data.copy()
    if isinstance(bins, int):
        edges = pd.qcut(binned[column], q=bins, duplicates="drop", retbins=True)[1]
        labels = _currency_bin_labels(edges)
        binned["_group"] = pd.cut(
            binned[column], bins=edges, labels=labels, include_lowest=True
        )
    else:
        edges, labels = bins
        binned["_group"] = pd.cut(
            binned[column],
            bins=edges,
            labels=labels,
            right=True,
            include_lowest=True,
        )
    return churn_rate_table(binned, "_group", sort_by_rate=False)


def print_churn_summary(title, summary):
    """Print a formatted churn-rate table."""
    print(f"\n{title}")
    print("-" * len(title))
    formatted = summary.copy()
    formatted["churn_rate_pct"] = formatted["churn_rate_pct"].map("{:.1f}%".format)
    print(formatted.to_string())


def plot_churn_rates(ax, summary, title, ordered=False):
    """Bar chart of churn rate per segment.

    ordered=True keeps the given order (used for binned numeric features so the
    bars read as a trend) and annotates each bar with its exact percentage.
    """
    # Wrap long labels onto multiple lines so they stay horizontal (no rotation).
    labels = [textwrap.fill(str(idx), width=12) for idx in summary.index]
    rates = summary["churn_rate_pct"]
    colors = plt.cm.Reds(np.linspace(0.35, 0.85, len(rates)))

    bars = ax.bar(labels, rates, color=colors)
    ax.set_ylabel("Churn rate (%)")
    ax.margins(y=0.15)
    ax.tick_params(axis="x", rotation=0)

    for bar, rate in zip(bars, rates):
        ax.annotate(
            f"{rate:.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(title)
    ax.axhline(
        df["Churn Value"].mean() * 100,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Overall churn rate",
    )
    ax.legend(loc="best", fontsize=8)


df_eda = add_internet_service_label(df)

overall_churn_pct = df["Churn Value"].mean() * 100
print(f"Overall churn rate: {overall_churn_pct:.1f}%  ({df['Churn Value'].sum():.0f} / {len(df)} customers)")

eda_summaries = []
eda_summaries.append(("Contract type", churn_rate_table(df_eda, "Contract")))
eda_summaries.append(("Internet Service", churn_rate_table(df_eda, "Internet Service")))
eda_summaries.append(("Payment Method", churn_rate_table(df_eda, "Payment Method")))

for title, summary in eda_summaries:
    print_churn_summary(title, summary)

for column, bins in CONTINUOUS_FEATURES.items():
    summary = churn_rate_by_bins(df_eda, column, bins)
    eda_summaries.append((column, summary))
    print_churn_summary(column, summary)

n_plots = len(eda_summaries)
n_cols = 2
n_rows = (n_plots + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
axes = axes.flatten()

PLOT_TITLE_SUFFIX = {"Monthly Charges": " ($)", "Total Charges": " ($)"}

for ax, (title, summary) in zip(axes, eda_summaries):
    plot_churn_rates(ax, summary, title + PLOT_TITLE_SUFFIX.get(title, ""))

for ax in axes[len(eda_summaries) :]:
    ax.axis("off")

fig.suptitle("Churn rate by customer segment", fontsize=14, y=1.01)
fig.tight_layout(pad=2.0, h_pad=3.0)
save_figure("churn_rate_by_segment.png")
maybe_show()

X = df_2.drop(columns=["Churn Value"])
y = df_2["Churn Value"]
churn_trait_ranking, overall_churn_pct = build_churn_likelihood_ranking(X, y)
plot_trait_churn_ranking(churn_trait_ranking, overall_churn_pct)


# ---------------------------------------------------------------------------
# Logistic Regression: predict Churn Value from df_2
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
X_test_scaled[CONTINUOUS_COLS] = scaler.transform(X_test[CONTINUOUS_COLS])

log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression")
print("=" * 40)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))
print("Confusion matrix (rows = actual, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------------
# Cross-validation: check whether logistic regression scores are stable
# ---------------------------------------------------------------------------

CV_FOLDS = 5
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_roc_auc = []
cv_accuracy = []
cv_recall = []

print(f"\n{CV_FOLDS}-fold cross-validation (Logistic Regression)")
print("=" * 40)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
    X_cv_train = X.iloc[train_idx]
    X_cv_val = X.iloc[val_idx]
    y_cv_train = y.iloc[train_idx]
    y_cv_val = y.iloc[val_idx]

    scaler_cv = StandardScaler()
    X_cv_train_scaled = X_cv_train.copy()
    X_cv_val_scaled = X_cv_val.copy()
    X_cv_train_scaled[CONTINUOUS_COLS] = scaler_cv.fit_transform(
        X_cv_train[CONTINUOUS_COLS]
    )
    X_cv_val_scaled[CONTINUOUS_COLS] = scaler_cv.transform(X_cv_val[CONTINUOUS_COLS])

    model_cv = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model_cv.fit(X_cv_train_scaled, y_cv_train)

    y_cv_pred = model_cv.predict(X_cv_val_scaled)
    y_cv_prob = model_cv.predict_proba(X_cv_val_scaled)[:, 1]

    fold_roc = roc_auc_score(y_cv_val, y_cv_prob)
    fold_acc = (y_cv_pred == y_cv_val).mean()
    fold_recall = recall_score(y_cv_val, y_cv_pred)

    cv_roc_auc.append(fold_roc)
    cv_accuracy.append(fold_acc)
    cv_recall.append(fold_recall)
    print(
        f"Fold {fold}: ROC-AUC={fold_roc:.3f}  "
        f"accuracy={fold_acc:.3f}  churn recall={fold_recall:.3f}"
    )

print(
    f"\nCV mean ROC-AUC:      {np.mean(cv_roc_auc):.3f} "
    f"(+/- {np.std(cv_roc_auc):.3f})"
)
print(
    f"CV mean accuracy:     {np.mean(cv_accuracy):.3f} "
    f"(+/- {np.std(cv_accuracy):.3f})"
)
print(
    f"CV mean churn recall: {np.mean(cv_recall):.3f} "
    f"(+/- {np.std(cv_recall):.3f})"
)
print(f"Single test split ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, CV_FOLDS + 1), cv_roc_auc, color="tab:orange", label="CV fold")
ax.axhline(
    roc_auc_score(y_test, y_prob),
    color="gray",
    linestyle="--",
    linewidth=1,
    label="Single test split",
)
ax.axhline(
    np.mean(cv_roc_auc),
    color="tab:red",
    linestyle=":",
    linewidth=1,
    label="CV mean",
)
ax.set_xlabel("Fold")
ax.set_ylabel("ROC-AUC")
ax.set_title(f"{CV_FOLDS}-fold cross-validation — ROC-AUC per fold")
ax.set_xticks(range(1, CV_FOLDS + 1))
ax.set_ylim(0, 1)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
save_figure("cv_roc_auc.png")
maybe_show()

# ---------------------------------------------------------------------------
# Model comparison: Random Forest & XGBoost (same train/test split)
# ---------------------------------------------------------------------------


def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    """Fit a classifier and return test-set metrics."""
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    return {
        "name": name,
        "model": model,
        "roc_auc": roc_auc_score(y_te, y_proba),
        "accuracy": (y_hat == y_te).mean(),
        "recall": recall_score(y_te, y_hat),
    }


def normalized_importance(values):
    """Scale importances so they sum to 1 for cross-model comparison."""
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total == 0:
        return values
    return values / total


lr_metrics = {
    "name": "Logistic Regression",
    "model": log_reg,
    "roc_auc": roc_auc_score(y_test, y_prob),
    "accuracy": (y_pred == y_test).mean(),
    "recall": recall_score(y_test, y_pred),
}

rf_metrics = evaluate_model(
    "Random Forest",
    RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    X_train,
    X_test,
    y_train,
    y_test,
)

if HAS_XGBOOST:
    boosting_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    boosting_name = "XGBoost"
else:
    boosting_model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    boosting_name = "Gradient Boosting"

boosting_metrics = evaluate_model(
    boosting_name, boosting_model, X_train, X_test, y_train, y_test
)

model_results = [lr_metrics, rf_metrics, boosting_metrics]
comparison = pd.DataFrame(model_results)[["name", "roc_auc", "accuracy", "recall"]]

print("\nModel comparison (same test split)")
print("=" * 40)
print(
    comparison.to_string(
        index=False,
        formatters={
            "roc_auc": "{:.3f}".format,
            "accuracy": "{:.3f}".format,
            "recall": "{:.3f}".format,
        },
    )
)
if not HAS_XGBOOST:
    print(
        "\nNote: xgboost is not installed, so HistGradientBoostingClassifier "
        "is used as the gradient-boosting comparison model."
    )

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
metric_labels = ["ROC-AUC", "Accuracy", "Churn recall"]
metric_cols = ["roc_auc", "accuracy", "recall"]
bar_colors = ["tab:blue", "tab:green", "tab:purple"]

for ax, metric_label, metric_col, color in zip(
    axes, metric_labels, metric_cols, bar_colors
):
    ax.bar(comparison["name"], comparison[metric_col], color=color)
    ax.set_title(metric_label)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

fig.suptitle("Model comparison on held-out test set", fontsize=14)
fig.tight_layout()
save_figure("model_comparison.png")
maybe_show()

def model_importance(metrics, X_te, y_te):
    """Return normalized feature importance for tree or linear models."""
    model = metrics["model"]
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        perm = permutation_importance(
            model, X_te, y_te, n_repeats=5, random_state=RANDOM_STATE
        )
        values = perm.importances_mean
    return pd.Series(normalized_importance(values), index=X.columns)


lr_importance = model_importance(lr_metrics, X_test_scaled, y_test)
rf_importance = model_importance(rf_metrics, X_test, y_test)
boost_importance = model_importance(boosting_metrics, X_test, y_test)

importance_df = pd.DataFrame(
    {
        "Logistic Regression": lr_importance,
        "Random Forest": rf_importance,
        boosting_name: boost_importance,
    }
)
top_features = (
    importance_df.max(axis=1).sort_values(ascending=False).head(12).index
)
importance_top = importance_df.loc[top_features]

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(top_features))
bar_height = 0.25
ax.barh(
    y_pos - bar_height,
    importance_top["Logistic Regression"],
    height=bar_height,
    label="Logistic Regression",
    color="tab:blue",
)
ax.barh(
    y_pos,
    importance_top["Random Forest"],
    height=bar_height,
    label="Random Forest",
    color="tab:green",
)
ax.barh(
    y_pos + bar_height,
    importance_top[boosting_name],
    height=bar_height,
    label=boosting_name,
    color="tab:purple",
)
ax.set_yticks(y_pos)
ax.set_yticklabels([textwrap.fill(str(name), width=30) for name in top_features])
ax.invert_yaxis()
ax.set_xlabel("Normalized importance")
ax.set_title("Top 12 features — agreement across models")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
save_figure("feature_importance_agreement.png")
maybe_show()

coefficients = pd.Series(log_reg.coef_[0], index=X.columns)
coefficients_by_importance = coefficients.sort_values(key=abs, ascending=False)
split_idx = len(coefficients_by_importance) // 2
most_important = coefficients_by_importance.iloc[:split_idx]
least_important = coefficients_by_importance.iloc[split_idx:].sort_values(
    key=abs, ascending=False
)


def plot_confusion_matrix(cm):
    """Standalone confusion matrix for the held-out test set."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Oranges")
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stayed", "Churned"])
    ax.set_yticklabels(["Stayed", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = cm[row, col]
            text_color = "white" if value > cm.max() * 0.55 else "black"
            ax.text(
                col, row, value, ha="center", va="center", color=text_color, fontsize=12
            )
    fig.tight_layout()
    save_figure("confusion_matrix.png")
    maybe_show()


def plot_coefficient_bars_on_ax(ax, coefficients, title, xlim):
    """Draw coefficient bars on an existing axes with a shared x-scale."""
    labels = [textwrap.fill(str(name), width=28) for name in coefficients.index]
    colors = ["tab:red" if value > 0 else "tab:blue" for value in coefficients]
    ax.barh(labels[::-1], coefficients.values[::-1], color=colors[::-1])
    ax.set_xlim(xlim)
    ax.set_xlabel("Coefficient (log-odds)")
    ax.set_title(title)
    ax.axvline(0, color="gray", linewidth=1)
    ax.tick_params(axis="y", labelsize=8)


def plot_coefficient_comparison(most_important, least_important, xlim):
    """Side-by-side coefficient plots with matching x-axis limits."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(6, 0.35 * len(most_important))),
        sharex=True,
    )
    plot_coefficient_bars_on_ax(
        axes[0],
        most_important,
        f"Most important features (top {len(most_important)} of {len(coefficients)})",
        xlim,
    )
    plot_coefficient_bars_on_ax(
        axes[1],
        least_important,
        f"Least important features (bottom {len(least_important)} of {len(coefficients)})",
        xlim,
    )
    fig.tight_layout()
    save_figure("coefficient_comparison.png")
    maybe_show()


cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)

xlim = (-coefficients.abs().max() * 1.08, coefficients.abs().max() * 1.08)
plot_coefficient_comparison(most_important, least_important, xlim)


# ---------------------------------------------------------------------------
# Feature importance ranking: consensus across all three models
# ---------------------------------------------------------------------------

TOP_CONSENSUS_FEATURES = 15

consensus_importance = (
    importance_df.mean(axis=1)
    .sort_values(ascending=False)
    .head(TOP_CONSENSUS_FEATURES)
)
consensus_labels = [
    readable_feature_name(feature) for feature in consensus_importance.index
]

print("\nFeature importance ranking (consensus across models)")
print("=" * 40)
consensus_table = pd.DataFrame(
    {
        "feature": consensus_labels,
        "avg_importance": consensus_importance.values,
        "logistic_regression": importance_df.loc[
            consensus_importance.index, "Logistic Regression"
        ].values,
        "random_forest": importance_df.loc[
            consensus_importance.index, "Random Forest"
        ].values,
        boosting_name: importance_df.loc[
            consensus_importance.index, boosting_name
        ].values,
    }
)
print(
    consensus_table.to_string(
        index=False,
        formatters={"avg_importance": "{:.3f}".format},
    )
)

fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.cm.Purples(np.linspace(0.45, 0.9, len(consensus_importance)))
bars = ax.barh(
    [textwrap.fill(label, width=32) for label in consensus_labels][::-1],
    consensus_importance.values[::-1],
    color=colors[::-1],
)
ax.set_xlabel("Average normalized importance")
ax.set_title("Feature importance ranking (consensus across models)")
ax.tick_params(axis="y", labelsize=8)

for bar, value in zip(bars, consensus_importance.values[::-1]):
    ax.text(
        bar.get_width() + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=8,
    )

fig.tight_layout()
save_figure("consensus_feature_importance.png")
maybe_show()