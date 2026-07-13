import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.dpi'] = 110
sns.set_style('whitegrid')

df = pd.read_csv('creditcard.csv')

# 1. Class imbalance plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
counts = df['Class'].value_counts()
axes[0].bar(['Legitimate', 'Fraud'], counts.values, color=['#4C72B0', '#C44E52'])
axes[0].set_yscale('log')
axes[0].set_ylabel('Count (log scale)')
axes[0].set_title('Class Distribution (log scale)')
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

axes[1].pie(counts.values, labels=['Legitimate (99.83%)', 'Fraud (0.17%)'],
            colors=['#4C72B0', '#C44E52'], autopct='%1.3f%%', startangle=90)
axes[1].set_title('Class Proportion')
plt.tight_layout()
plt.savefig('plots/01_class_imbalance.png', bbox_inches='tight')
plt.close()

# 2. Amount distribution by class
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x='Class', y='Amount', ax=axes[0])
axes[0].set_yscale('log')
axes[0].set_xticklabels(['Legitimate', 'Fraud'])
axes[0].set_title('Transaction Amount by Class (log scale)')

sns.histplot(df[df.Class==0]['Amount'], bins=50, color='#4C72B0', label='Legitimate',
             stat='density', ax=axes[1], log_scale=(True, False))
sns.histplot(df[df.Class==1]['Amount'], bins=50, color='#C44E52', label='Fraud',
             stat='density', ax=axes[1], log_scale=(True, False))
axes[1].set_title('Amount Distribution (density)')
axes[1].legend()
plt.tight_layout()
plt.savefig('plots/02_amount_distribution.png', bbox_inches='tight')
plt.close()

# 3. Time pattern
df['Hour'] = (df['Time'] // 3600) % 24
fig, ax = plt.subplots(figsize=(11, 4.5))
fraud_by_hour = df[df.Class==1].groupby('Hour').size()
legit_by_hour = df[df.Class==0].groupby('Hour').size() / 1000  # scaled
ax2 = ax.twinx()
ax.bar(fraud_by_hour.index, fraud_by_hour.values, color='#C44E52', alpha=0.7, label='Fraud count')
ax2.plot(legit_by_hour.index, legit_by_hour.values, color='#4C72B0', marker='o', label='Legit count (thousands)')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Fraud transaction count', color='#C44E52')
ax2.set_ylabel('Legit transaction count (thousands)', color='#4C72B0')
ax.set_title('Transaction Volume by Hour of Day')
plt.tight_layout()
plt.savefig('plots/03_time_pattern.png', bbox_inches='tight')
plt.close()

# 4. Correlation of V-features with Class
corrs = df.drop(columns=['Hour']).corr()['Class'].drop('Class').sort_values()
fig, ax = plt.subplots(figsize=(9, 8))
colors = ['#C44E52' if x < 0 else '#4C72B0' for x in corrs.values]
ax.barh(corrs.index, corrs.values, color=colors)
ax.set_title('Feature Correlation with Fraud (Class)')
ax.set_xlabel('Pearson correlation')
plt.tight_layout()
plt.savefig('plots/04_feature_correlation.png', bbox_inches='tight')
plt.close()

print("EDA plots saved.")
print("\nTop 5 positively correlated features:", corrs.tail(5).to_dict())
print("Top 5 negatively correlated features:", corrs.head(5).to_dict())
print("\nAmount stats by class:")
print(df.groupby('Class')['Amount'].describe())