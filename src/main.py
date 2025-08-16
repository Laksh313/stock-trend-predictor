#the main file to run everything in order
#possibly add a UI later
from data_fetch import fetch_stock_data
from data_preprocessing import load_and_clean, engineer_features
from model_training import train_model

data_path_input = input("Enter Stock Symbol:").upper()
data_path = f'data/{data_path_input}_data.csv'

model_path = 'model/stock_predictor.pkl'

fetch_stock_data(data_path_input, "2023-01-01", "2024-01-01")

df = load_and_clean(data_path)
#df = engineer_features(df)

train_model(df, model_path)