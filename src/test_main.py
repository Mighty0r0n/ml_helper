#from sklearnex import unpatch_sklearn, patch_sklearn
#patch_sklearn()  # May cause troubles on WSL
import config
from sklearn.model_selection import (GridSearchCV, train_test_split, cross_validate)


import utils
from logging_assistance.logging_helper import logger
from logging_assistance.logging_helper import setup_logging, generate_run_directories


def main(data_path: str, target: str, model, model_param_grid: dict, test_size: float, random_state: int, cv: int):
    # Setup the logging
    setup_logging('../configs/logging_config.json')

    # Create the run directories
    main_dir, model_dir, plot_dir, log_dir = generate_run_directories(log_name=model.__class__.__name__,
        tag=f"{model.__class__.__name__}_placeholder")

    # Load the data
    data = utils.load_data(data_path, decimal=".")

    # Split the data into features and target
    X = data.drop(target, axis=1)
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create a grid search
    grid_search = GridSearchCV(model, model_param_grid, cv=cv, verbose=3, return_train_score=True, n_jobs=-1, error_score='raise').fit(
        X_train, y_train)

    # Fit the grid search to the data
    best_estimator = grid_search.best_estimator_
    utils.save_model(model=best_estimator, path=model_dir + f"{model.__class__.__name__}.pkl")
    utils.save_params(logdir=main_dir + "/models/", filename=model.__class__.__name__, params=grid_search.best_params_)

    # Print the best parameters
    logger.info(grid_search.best_params_)

    # Print the best score
    logger.info(grid_search.best_score_)

    # Predict the target
    y_pred = grid_search.predict(X_test)

    logger.info(utils.print_regression_metrics(y_test, y_pred))

    # Print the cross validation scores
    logger.info(cross_validate(best_estimator, X, y, cv=cv, scoring=('r2', 'max_error')))


if __name__ == '__main__':
    main(data_path='../test_data/testdata.csv', target='Target', model=config.rf_regressor,
         model_param_grid=config.rf_regressor_param_grid, test_size=0.2, random_state=42, cv=5)
