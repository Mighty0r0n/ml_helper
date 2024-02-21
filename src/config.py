from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()

rf_regressor_param_grid = {
    'n_estimators': [100],
    'min_samples_split': [3],
    'min_samples_leaf': [1],
}
