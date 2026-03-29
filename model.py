import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

def train_and_predict_advanced(df, target='PTS'):
    # Dynamically select the correct rolling average based on the target
    target_roll = f'{target}_ROLL_5'
    
    features = [
        target_roll, 'MIN_ROLL_5', 'MIN_STD_ROLL_5', 'SHOT_ATT_ROLL_5', 
        'TS_ROLL_5', 'GAMES_LAST_7_DAYS', 'REST_DAYS', 'IS_B2B', 
        'HOME_GAME', 'USG_ROLL_5', 'INTERACTION_USG_DEF', 'INTERACTION_MIN_PACE'
    ]
    
    X = df[features][:-1]
    y = df[target][1:]
    
    tscv = TimeSeriesSplit(n_splits=3)
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        
    latest_features = df[features].iloc[-1:]
    pred = model.predict(latest_features)
    
    return pred[0], model, features