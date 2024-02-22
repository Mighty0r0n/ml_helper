from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()

rf_regressor_param_grid = {'n_estimators': [1000], 'min_samples_split': [5], 'min_samples_leaf': [5], }
