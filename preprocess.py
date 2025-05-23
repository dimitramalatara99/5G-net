import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
import matplotlib.pyplot as plt

df = pd.read_csv("new.csv")

# data types- no need for conversion
df['Latitude'].dtype
print (df.dtypes)

pathloss_binned = []

# min-max normalization
scaler = MinMaxScaler()
values = df.columns.difference(["PathLoss(db)"])  # select all columns except path-loss
df[values] = scaler.fit_transform(df[values])

# binning to make it stratified for the cv (later)
num_bins = 7
binning = KBinsDiscretizer (num_bins, encode="ordinal", strategy="quantile")
pathloss_binned = binning.fit_transform(df[["PathLoss(db)"]]).flatten()

# display range of the predictors
df.plot(kind='box', subplots=True, layout=(3, 3), figsize=(8, 8), sharex=False, sharey=False)
plt.suptitle("Box Plots for Numerical Columns")
plt.show()

# visualization after min max
df[values].plot(kind='box', subplots=True, layout=(3, 3), figsize=(8, 8))
plt.suptitle("Box Plots After MinMax")
plt.show()

# bins distribution
print("\nBin distribution of PathLoss:")
print(pd.Series(pathloss_binned).value_counts().sort_index())

# new new dataset
df.to_csv("new_new.csv", index=False)
print("\nProcessed dataset saved as 'new_new.csv'")

sns.histplot(df["PathLoss(db)"], bins=30, kde=True)
plt.title("PathLoss(db) Distribution")
plt.show()