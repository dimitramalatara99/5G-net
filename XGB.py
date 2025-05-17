import numpy as np
import pandas as pd
from numpy import absolute
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, KFold
import optuna
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, cv, plot_importance
import seaborn as sns
from sklearn.inspection import permutation_importance
from matplotlib.colors import ListedColormap

df = pd.read_csv("new_new.csv")

X = df.drop(columns=['PathLoss(db)'])  # select all columns except path-loss
y = df['PathLoss(db)']

# training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

n_samples = 100
kf = KFold(n_splits=10, shuffle=False)

# 0: unused, 1: train, 2: validation
cv_visual = np.zeros((10, n_samples))

# Συμπλήρωση του πίνακα με 1 (train) και 2 (test) για κάθε fold
for fold_idx, (train_index, test_index) in enumerate(kf.split(range(n_samples))):
    cv_visual[fold_idx, train_index] = 1
    cv_visual[fold_idx, test_index] = 2

plt.figure(figsize=(12, 4))
colors = ["lightblue", "lightgray", "violet"]
custom_cmap = ListedColormap(colors)
plt.imshow(cv_visual, interpolation='nearest', cmap=custom_cmap)
plt.title("10-Fold Cross-Validation Split Visualization (Before Prediction)")
plt.xlabel("Samples")
plt.ylabel("Fold")
cbar = plt.colorbar(ticks=[0, 1, 2], label='Role')
cbar.ax.set_yticklabels(['Unused', 'Train', 'Validation'])
plt.tight_layout()
plt.show()


params = {'objective': 'reg:squarederror',
         'max_depth': 8,  # of the tree 10
         'alpha': 10,  # in case of overfiting L1 Regularization 2
         'learning_rate': 0.1,  # slow learning
         'n_estimators': 210,  # number of trees 210
         'lambda': 5, #L2 Regularization 3, 0.8
         'gamma': 0.07 #check overfitting 0.03
         }
xgb_reg = XGBRegressor(**params)
xgb_reg.fit(X_train, y_train)


# test set predictions(smaller)
y_pred = xgb_reg.predict(X_test)
print(f"error rate for the test data", mean_absolute_error(y_test, y_pred))  # difference between prediction/actual


# trained set predictions
y_pred_train = xgb_reg.predict(X_train)
print("error rate for the trained data", mean_absolute_error(y_train, y_pred_train))

plot_importance(xgb_reg)
plt.title("Feature Importance (XGBoost Regressor)")
plt.rcParams['figure.figsize'] = [8, 4]
plt.show(),

# correlation matrix for explanation Feature Importance
correlation_matrix = df.corr()
print(correlation_matrix)

# Print heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

result = permutation_importance(xgb_reg, X_test, y_test, scoring='neg_mean_absolute_error', n_repeats=10, random_state=42)
perm_importance = pd.DataFrame({"Feature": X.columns, "Importance": result.importances_mean})
perm_importance = perm_importance.sort_values(by="Importance", ascending=False)

print(perm_importance)

# Feature importance από το XGBoost
importance = xgb_reg.feature_importances_
features = X.columns

# Correlation με το target variable
correlations = df.corr()["PathLoss(db)"].drop("PathLoss(db)")


# model evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)  # , as_pandas=True
scores = cross_val_score(xgb_reg, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print(scores[:5])

# make scores positive
scores = absolute(scores)
print('MAE evaluation: %.3f (%.3f)' % (scores.mean(), scores.std()))


# optuna framework for tuning hyperparameters
def objective(trial):
   params_opt = {
       'objective': 'reg:squarederror',
       'n_estimators': trial.suggest_int('n_estimators', 50, 300),
       'max_depth': trial.suggest_int('max_depth', 3, 12),
       'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
       'alpha': trial.suggest_float('alpha', 0, 15),
       'lambda': trial.suggest_float('lambda', 0, 15),  # L2 regularization
       'gamma': trial.suggest_float('gamma', 0, 10),
   }

   model = XGBRegressor(**params_opt)

   # 10 fold cv named yolo because that's us
   yolo = cross_val_score(model, X_train, y_train,
                          scoring='neg_mean_absolute_error',
                          cv=10, n_jobs=-1)

   return -np.mean(yolo)  # makes the objective smaller


# run optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, timeout=130)  # 30 trials or 130 seconds


print("Best :", study.best_params)
print("Best CV MAE:", study.best_value)

mean_actual = y_test.mean()
mae_percentage = (2.370 / mean_actual) * 100

print(f"MAE as a percentage: {mae_percentage:.2f}%")

# tarin final model with best params
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)


# test set evaluation
y_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print("Test MAE:", test_mae)

# Υπολογισμός MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.3f}")

# std_y = np.std(df['PathLoss(db)'], ddof=1)  # ddof=1 για να υπολογιστεί σωστά για δείγμα
# print("Τυπικό σφάλμα του PathLoss:", std_y)


explainer = shap.Explainer(xgb_reg, X_train)
shap_values = explainer(X_test)


# visualize the predictions
shap.plots.beeswarm(shap_values)
shap.plots.waterfall(shap_values[5])


train_errors = []
test_errors = []


n_trees = range(10, 200, 10)

best_params = study.best_params.copy()
best_params.pop("n_estimators", None)

for n in n_trees:
    model = XGBRegressor(n_estimators=n, **best_params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_errors.append(train_mae)
    test_errors.append(test_mae)


plt.figure(figsize=(8, 5))
plt.plot(n_trees, train_errors, label="Training Error", marker='o')
plt.plot(n_trees, test_errors, label="Test Error", marker='s')
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Overfitting Check: Train vs Test Error")
plt.legend()
plt.grid()
plt.show()



kf = KFold(n_splits=10, shuffle=True, random_state=42)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 γραμμές, 5 στήλες για τα 10 folds
axes = axes.flatten()

for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBRegressor(**study.best_params)  # Χρήση των καλύτερων παραμέτρων από Optuna
    model.fit(X_train_fold, y_train_fold)

    y_pred_fold = model.predict(X_test_fold)

    # Scatter plot
    axes[i].scatter(y_test_fold, y_pred_fold, alpha=0.5)
    axes[i].plot([min(y_test_fold), max(y_test_fold)], [min(y_test_fold), max(y_test_fold)], 'r--')  # Διαγώνια γραμμή y=x
    axes[i].set_title(f"Fold {i+1}")
    axes[i].set_xlabel("Actual")
    axes[i].set_ylabel("Predicted")

plt.tight_layout()
plt.show()
