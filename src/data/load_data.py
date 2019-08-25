import pandas as pd

def load_data():

	df = pd.read_csv("../data/raw/train.csv")

	df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0])

	feats = df.corrwith(df['SalePrice']).sort_values(ascending=False).index[1:10]

	X = df[feats]
	y = df['SalePrice']

	return X, y