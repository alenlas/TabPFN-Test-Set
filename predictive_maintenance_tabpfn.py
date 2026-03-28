"""
RobotOps Copilot — Predictive Maintenance Model
Dataset: AI4I 2020 Predictive Maintenance (UCI) — auto-downloads
Primary model: XGBoost (fast, no auth needed, great SHAP support)
Optional: TabPFN (requires HuggingFace login)

Install:
    pip install xgboost ucimlrepo scikit-learn shap pandas numpy matplotlib

Run:
    python predictive_maintenance_tabpfn.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ── Optional SHAP ────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed — run: pip install shap")

# ── Optional TabPFN (gated — requires HuggingFace login) ────────────────────
TABPFN_AVAILABLE = False
# Uncomment below if you've run `hf auth login` and accepted terms at:
# https://huggingface.co/Prior-Labs/tabpfn_2_5
# try:
#     from tabpfn import TabPFNClassifier
#     TABPFN_AVAILABLE = True
# except Exception as e:
#     print(f"⚠️  TabPFN unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("\n📦 Loading AI4I 2020 Predictive Maintenance Dataset...")

dataset = fetch_ucirepo(id=601)
X_raw = dataset.data.features.copy()
y_raw = dataset.data.targets.copy()

print(f"✅ Loaded {len(X_raw):,} rows, {X_raw.shape[1]} features")
print(f"   Failure rate: {y_raw['Machine failure'].mean():.1%}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔧 Engineering features...")

df = X_raw.copy()
print(f"   Columns found: {df.columns.tolist()}\n")

def find_col(df, keywords):
    kw = [k.lower() for k in keywords]
    for col in df.columns:
        if all(k in col.lower() for k in kw):
            return col
    raise KeyError(f"No column matching {keywords} in {df.columns.tolist()}")

col_type   = find_col(df, ["type"])
col_air    = find_col(df, ["air", "temp"])
col_proc   = find_col(df, ["process", "temp"])
col_rpm    = find_col(df, ["rotational"])
col_torque = find_col(df, ["torque"])
col_wear   = find_col(df, ["tool", "wear"])

print(f"   Mapped → air='{col_air}' | proc='{col_proc}' | rpm='{col_rpm}' | torque='{col_torque}' | wear='{col_wear}'")

le = LabelEncoder()
df["type_encoded"] = le.fit_transform(df[col_type])
df.drop(columns=[col_type], inplace=True)

id_cols = [c for c in df.columns if any(x in c.lower() for x in ["uid", "product id", "productid", "product_id"])]
df.drop(columns=id_cols, inplace=True, errors="ignore")

df["temp_diff"]   = df[col_proc] - df[col_air]
df["power_kw"]    = df[col_torque] * df[col_rpm] * (2 * np.pi / 60) / 1000
df["wear_torque"] = df[col_wear] * df[col_torque]
df["high_torque"] = (df[col_torque] > 50).astype(int)
df["low_speed"]   = (df[col_rpm] < 1380).astype(int)
df["wear_high"]   = (df[col_wear] > 180).astype(int)

feature_names = df.columns.tolist()
X = df.values
y = y_raw["Machine failure"].values

print(f"✅ Feature matrix: {X.shape}  |  Features: {feature_names}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=1000, random_state=42, stratify=y
)

print(f"\n🔀 Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"   Train failure rate: {y_train.mean():.1%}  |  Test: {y_test.mean():.1%}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. TRAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n🚀 Training model...")

if TABPFN_AVAILABLE:
    model = TabPFNClassifier(device="cpu")
    model_name = "TabPFN"
else:
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="logloss",
    )
    model_name = "XGBoost"

model.fit(X_train, y_train)
print(f"✅ {model_name} trained!")


# ═══════════════════════════════════════════════════════════════════════════
# 5. EVALUATE
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n📊 Evaluation — {model_name}:")
print("─" * 50)

y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Failure', 'Failure'])}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Predictive Maintenance — {model_name}", fontsize=14, fontweight="bold")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["No Failure", "Failure"]).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix")

axes[1].hist(y_prob[y_test == 0], bins=40, alpha=0.6, label="No Failure", color="#2196F3")
axes[1].hist(y_prob[y_test == 1], bins=40, alpha=0.6, label="Failure",    color="#F44336")
axes[1].axvline(0.5, color="black", linestyle="--", label="Threshold 0.5")
axes[1].set_xlabel("Predicted Failure Probability")
axes[1].set_ylabel("Count")
axes[1].set_title("Risk Score Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig("maintenance_evaluation.png", dpi=150)
plt.show()
print("💾 Saved: maintenance_evaluation.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════
if SHAP_AVAILABLE:
    print("\n🔍 Computing SHAP values...")
    explainer   = shap.Explainer(model, pd.DataFrame(X_train, columns=feature_names))
    shap_values = explainer(pd.DataFrame(X_test[:300], columns=feature_names))

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("Feature Impact on Failure Prediction (SHAP)", fontsize=13)
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150)
    plt.show()
    print("💾 Saved: shap_beeswarm.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. TOP MACHINES AT RISK
# ═══════════════════════════════════════════════════════════════════════════
def risk_label(p):
    if p >= 0.80: return "🔴 CRITICAL"
    if p >= 0.50: return "🟠 HIGH"
    if p >= 0.25: return "🟡 MEDIUM"
    return "🟢 LOW"

results_df = pd.DataFrame(X_test, columns=feature_names)
results_df["machine_id"]     = [f"ROBOT-{i:04d}" for i in range(len(X_test))]
results_df["failure_risk_%"] = (y_prob * 100).round(1)
results_df["risk_tier"]      = [risk_label(p) for p in y_prob]
results_df["actual_failure"] = y_test

print("\n🏆 Top 10 Machines at Highest Failure Risk:")
print("─" * 60)
top10 = (results_df
         .sort_values("failure_risk_%", ascending=False)
         .head(10)[["machine_id", "failure_risk_%", "risk_tier", "actual_failure"]])
print(top10.to_string(index=False))

print("\n📋 Fleet Risk Summary:")
for tier, count in results_df["risk_tier"].value_counts().items():
    print(f"   {tier}: {count:,} machines ({count/len(results_df):.1%})")


# ═══════════════════════════════════════════════════════════════════════════
# 8. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
results_df.to_csv("maintenance_predictions.csv", index=False)
print(f"\n💾 Saved: maintenance_predictions.csv ({len(results_df):,} rows)")
print("✅ Done! Next: streamlit run dashboard.py")