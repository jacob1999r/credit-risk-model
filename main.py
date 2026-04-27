import pandas as pd

#load
df = pd.read_csv(
    #dataset is Lendind Club accepted loan data from 2007 to 2018
    #available on Kaggle https://www.kaggle.com/datasets/wordsforthewise/lending-club?select=accepted_2007_to_2018Q4.csv.gz
    'data/accepted_2007_to_2018Q4.csv',
    low_memory=False
)
#sample
df = df.sample(n=100000, random_state=42)

#check attributes
print(df.shape)       # number of rows and columns
print(df.head())      # first few rows
print(df.info())      # column names, data types, null counts
print(df.describe())  # summary statistics