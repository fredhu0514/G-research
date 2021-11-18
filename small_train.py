import pandas as pd

df = pd.read_csv("train.csv")

small_df = df.head(1024)

small_df.to_csv("small_train.csv")