import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "../data/tourism_5000_rows.csv")
df = pd.read_csv(data_path)

# Create emission level classification
total_emissions = df['total_trip_emissions_kgCO2']
percentile_33 = total_emissions.quantile(0.33)
percentile_67 = total_emissions.quantile(0.67)

def classify_emission(value):
    if value <= percentile_33:
        return 'low'
    elif value <= percentile_67:
        return 'medium'
    else:
        return 'high'

df['emission_level'] = df['total_trip_emissions_kgCO2'].apply(classify_emission)

# Remove data leakage columns
leaky_columns = [
    'emission_level', 'total_trip_emissions_kgCO2', 'transport_emissions_kgCO2',
    'accommodation_elec_kgCO2', 'accommodation_gen_kgCO2', 'festival_gen_emissions_kgCO2',
    'pilgrimage_emissions_kgCO2', 'food_emissions_kgCO2', 'rice_emissions_kgCO2',
    'waste_emissions_kgCO2', 'plastic_emissions_kgCO2', 'record_id', 'trip_start_date'
]

feature_df = df.drop(columns=[col for col in leaky_columns if col in df.columns])
cat_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)
feature_df = feature_df.fillna(0)

X = feature_df
y = df['emission_level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training models...")

# Train models
lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled))

dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

rf_model = RandomForestClassifier(
    n_estimators=800, random_state=42, max_depth=20,
    max_features='sqrt', min_samples_leaf=2, min_samples_split=4,
    n_jobs=-1, class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"Decision Tree Accuracy: {dt_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============================================
# PLOT 1: Model Comparison Bar Chart
# ============================================
models = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest']
accuracies = [lr_acc, dt_acc, rf_acc]
colors = ['#3498db', '#e74c3c', '#2ecc71']

ax1 = axes[0]
bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.annotate(f'{acc:.2%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=14, fontweight='bold')

# Highlight the best model
best_idx = accuracies.index(max(accuracies))
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_title('Model Comparison: Accuracy', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(0.75, 0.90)
ax1.axhline(y=max(accuracies), color='gold', linestyle='--', alpha=0.7, label='Best Accuracy')
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# Add annotation for best model
ax1.annotate('🏆 Best Model', 
            xy=(best_idx, max(accuracies)), 
            xytext=(best_idx + 0.3, max(accuracies) - 0.02),
            fontsize=11, color='green', fontweight='bold')

# ============================================
# PLOT 2: Feature Importance (Random Forest)
# ============================================
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

# Get top 15 features
top_features = feature_importance.tail(15)

ax2 = axes[1]
bars2 = ax2.barh(range(len(top_features)), top_features['Importance'], 
                 color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features))))

# Clean feature names for display
clean_names = []
for name in top_features['Feature']:
    # Shorten long feature names
    if len(name) > 25:
        name = name[:22] + '...'
    clean_names.append(name)

ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(clean_names, fontsize=9)
ax2.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax2.set_title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, imp) in enumerate(zip(bars2, top_features['Importance'])):
    ax2.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=8)

plt.tight_layout()

# Save the figure
output_path = os.path.join(os.path.dirname(__file__), "../model/model_comparison_charts.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ Charts saved to: {output_path}")

# Show the plot
plt.show()

print("\n📊 Visualization complete!")
