import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                              roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

plt.rcParams['figure.dpi'] = 110
sns.set_style('whitegrid')
np.random.seed(42)

df = pd.read_csv('creditcard.csv')

# ---- Feature engineering ----
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Hour'] = (df['Time'] // 3600) % 24
df['Time_scaled'] = StandardScaler().fit_transform(df[['Time']])

features = [c for c in df.columns if c.startswith('V')] + ['Amount_scaled', 'Time_scaled', 'Hour']
X = df[features]
y = df['Class']

# Stratified split preserves fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, fraud rate: {y_train.mean():.5f}")
print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean():.5f}")

# ---- Handle imbalance: SMOTE on training data only ----
smote = SMOTE(random_state=42, sampling_strategy=0.1)  # bring fraud to 10% of majority, not full 50/50
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_res.shape}, fraud rate: {y_train_res.mean():.5f}")

results = {}
models = {}

def evaluate(name, model, X_te, y_te, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_te, y_scores)
    ap = average_precision_score(y_te, y_scores)
    roc_auc = roc_auc_score(y_te, y_scores)
    y_pred = (y_scores >= 0.5).astype(int)
    cm = confusion_matrix(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    results[name] = {
        'PR_AUC': ap, 'ROC_AUC': roc_auc, 'F1': f1,
        'Precision': prec, 'Recall': rec,
        'confusion_matrix': cm.tolist()
    }
    return precision, recall, ap

# ---- Model 1: Logistic Regression (trained on SMOTE'd data) ----
t0 = time.time()
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)
lr_scores = lr.predict_proba(X_test)[:, 1]
evaluate('Logistic Regression', lr, X_test, y_test, lr_scores)
models['Logistic Regression'] = lr
print(f"LR trained in {time.time()-t0:.1f}s")

# ---- Model 2: Random Forest (class_weight balanced) ----
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
rf_scores = rf.predict_proba(X_test)[:, 1]
evaluate('Random Forest', rf, X_test, y_test, rf_scores)
models['Random Forest'] = rf
print(f"RF trained in {time.time()-t0:.1f}s")

# ---- Model 3: XGBoost (scale_pos_weight for imbalance, trained on original data) ----
t0 = time.time()
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, eval_metric='aucpr',
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
evaluate('XGBoost', xgb_model, X_test, y_test, xgb_scores)
models['XGBoost'] = xgb_model
print(f"XGB trained in {time.time()-t0:.1f}s")

# ---- Save results ----
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

for name, r in results.items():
    print(f"\n{name}:")
    print(f"  PR-AUC: {r['PR_AUC']:.4f}  ROC-AUC: {r['ROC_AUC']:.4f}")
    print(f"  Precision: {r['Precision']:.4f}  Recall: {r['Recall']:.4f}  F1: {r['F1']:.4f}")
    print(f"  Confusion Matrix: {r['confusion_matrix']}")

# ---- Plot 1: Precision-Recall curves ----
fig, ax = plt.subplots(figsize=(8, 6))
score_dict = {'Logistic Regression': lr_scores, 'Random Forest': rf_scores, 'XGBoost': xgb_scores}
colors = {'Logistic Regression': '#4C72B0', 'Random Forest': '#55A868', 'XGBoost': '#C44E52'}
for name, scores in score_dict.items():
    precision, recall, _ = precision_recall_curve(y_test, scores)
    ap = results[name]['PR_AUC']
    ax.plot(recall, precision, label=f'{name} (AP={ap:.3f})', color=colors[name], linewidth=2)
baseline = y_test.mean()
ax.axhline(baseline, linestyle='--', color='gray', label=f'Random baseline ({baseline:.4f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves (Test Set)')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig('plots/05_precision_recall_curves.png', bbox_inches='tight')
plt.close()

# ---- Plot 2: ROC curves ----
fig, ax = plt.subplots(figsize=(8, 6))
for name, scores in score_dict.items():
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = results[name]['ROC_AUC']
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=colors[name], linewidth=2)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves (Test Set)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/06_roc_curves.png', bbox_inches='tight')
plt.close()

# ---- Plot 3: Confusion matrices ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, r) in zip(axes, results.items()):
    cm = np.array(r['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/07_confusion_matrices.png', bbox_inches='tight')
plt.close()

# ---- Plot 4: Feature importance ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
rf_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).head(12)
xgb_imp = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False).head(12)
axes[0].barh(rf_imp.index[::-1], rf_imp.values[::-1], color='#55A868')
axes[0].set_title('Random Forest - Top 12 Feature Importances')
axes[1].barh(xgb_imp.index[::-1], xgb_imp.values[::-1], color='#C44E52')
axes[1].set_title('XGBoost - Top 12 Feature Importances')
plt.tight_layout()
plt.savefig('plots/08_feature_importance.png', bbox_inches='tight')
plt.close()

# ---- Precision/Recall vs threshold (XGBoost) ----
precision, recall, thresholds = precision_recall_curve(y_test, xgb_scores)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, precision[:-1], label='Precision', color='#4C72B0')
ax.plot(thresholds, recall[:-1], label='Recall', color='#C44E52')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Score')
ax.set_title('XGBoost: Precision & Recall vs Threshold')
ax.legend()
ax.axvline(0.5, linestyle=':', color='gray', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/09_threshold_tradeoff.png', bbox_inches='tight')
plt.close()

print("\nAll plots saved. Modeling complete.")
best_model_name = max(results, key=lambda k: results[k]['PR_AUC'])
print(f"\nBest model by PR-AUC: {best_model_name}")