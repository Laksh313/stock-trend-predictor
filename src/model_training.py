#train the model here (and later make changes to improve it)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(df, model_path):
    x = df.drop(columns=['Target'])
    x = x.select_dtypes(include=["float64", "int64"])
    y = df['Target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    import os
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")