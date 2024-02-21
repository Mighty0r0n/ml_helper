# from sklearnex import patch_sklearn
# patch_sklearn()  # Still receiving NaNs -> Why??
from sklearn.model_selection import (GridSearchCV, train_test_split, cross_validate)
from src import config, logging_helper
import utils


def main(data_path: str,
         target: str,
         model,
         model_param_grid: dict,
         test_size: float,
         random_state: int,
         cv: int):

    # Setup the logging
    logging_helper.setup_logging('../configs/logging_config.json')

    # Create the run directories
    error_dir, main_dir, model_dir = logging_helper.generate_run_directories(
        tag=f"{model.__class__.__name__}_placeholder"
    )

    # Load the data
    data = utils.load_data(data_path, decimal=".")

    # Split the data into features and target
    X = data.drop(target, axis=1)
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    # Create a grid search
    grid_search = GridSearchCV(model,
                               model_param_grid,
                               cv=cv,
                               verbose=3,
                               return_train_score=True,
                               n_jobs=-1).fit(X_train, y_train)

    # Fit the grid search to the data
    best_estimator = grid_search.best_estimator_
    utils.save_model(model=best_estimator, path=model_dir + f"{model.__class__.__name__}.pkl")
    utils.save_params(logdir=main_dir + "/models/", filename=model.__class__.__name__,
                      params=grid_search.best_params_)

    # Print the best parameters
    print(grid_search.best_params_)

    # Print the best score
    print(grid_search.best_score_)

    # Predict the target
    y_pred = grid_search.predict(X_test)

    logging_helper.print_regression_metrics(y_test, y_pred)

    # Print the cross validation scores
    print(cross_validate(best_estimator, X, y, cv=cv, scoring=('r2',
                                                                           'max_error')
                         ))




if __name__ == '__main__':
    main(data_path='../test_data/testdata.csv',
         target='Target',
         model=config.rf_regressor,
         model_param_grid=config.rf_regressor_param_grid,
         test_size=0.2,
         random_state=42,
         cv=5)