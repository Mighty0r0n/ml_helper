from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()

rf_regressor_param_grid = {
    'n_estimators': [100, 200, 300],
    'min_samples_split': [3, 4, 5],
    'min_samples_leaf': [1, 2, 3],
}
