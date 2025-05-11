import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras import layers, models
import optuna
import shap
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

df = pd.read_csv("new_new.csv")
X = df.drop(columns=['PathLoss(db)', 'PathLoss_binned'], errors='ignore')
y = df['PathLoss(db)']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# optuna objective function
def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_int("units", 16, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_maes = []

    for train_idx, val_idx in kf.split(X_scaled):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y.values[train_idx], y.values[val_idx]

        # MLP
        model = models.Sequential()
        model.add(layers.Input(shape=(X_scaled.shape[1],)))
        for _ in range(n_layers):
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_absolute_error'
        )

        model.fit(X_tr, y_tr,
                  epochs=50,
                  batch_size=32,
                  verbose=0)

        # validation
        preds = model.predict(X_val, verbose=0).flatten()
        fold_maes.append(mean_absolute_error(y_val, preds))

    # mae across all folds
    return float(np.mean(fold_maes))


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best hyperparameters:", study.best_params)
print("Best 10-fold CV MAE:", study.best_value)

# final model with best hyperparam
best = study.best_params
final_model = models.Sequential()
final_model.add(layers.Input(shape=(X_scaled.shape[1],)))
for _ in range(best['n_layers']):
    final_model.add(layers.Dense(best['units'], activation='relu'))
    final_model.add(layers.Dropout(best['dropout']))
final_model.add(layers.Dense(1))
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best['lr']),
    loss='mean_absolute_error'
)

final_model.fit(X_scaled, y,
                epochs=50,
                batch_size=32,
                verbose=1)

# # 6. Quick check of training MAE
# train_preds = final_model.predict(X_scaled).flatten()
# print(f"MAE on full data: {mean_absolute_error(y, train_preds):.4f}")

y_pred = final_model.predict(X_scaled).flatten()

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mape = np.mean(np.abs((y - y_pred) / y)) * 100
r2 = r2_score(y, y_pred)

print(f"MAE:   {mae:.4f} dB")
print(f"RMSE:  {rmse:.4f} dB")
print(f"MAPE:  {mape:.2f}%")
print(f"R²:    {r2:.4f}")

# explainer for 100 first samples
explainer = shap.Explainer(final_model, X, feature_names=list(X.columns))
shap_values = explainer(X.iloc[:100])  # returns an Explanation object

shap.plots.beeswarm(shap_values)  # global importance
shap.plots.waterfall(shap_values[1])  # only 1 sample