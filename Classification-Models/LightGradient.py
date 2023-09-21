import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, classification_report

class LightGradient:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # Training
        self.X_train = X_train
        self.y_train = y_train

        # Validation
        self.X_val = X_val
        self.y_val = y_val

        # Testing
        self.X_test = X_test
        self.y_test = y_test

        # Setting default params
        self.default_params = {
            'learning_rate': 0.05,
            'n_estimators': 100,
            'early_stopping_rounds': 10,
        }
    
    # LightGBM with default params
    def LightGBM_Classic(self):
        """
        Returns:
        lgbm_model: The classic LightGBM model trained on the passed data.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.
        """
        lgb_model = lgb.LGBMClassifier(**self.default_params)
        # Train model
        lgb_model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)])
        
        # Validation
        pred_val = lgb_model.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = lgb_model.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test) 

        # Printing the results
        print(f"LightGBM Classifier's validation accuracy is {accuracy_val}")
        print("-" * 70)
        print(f"LightGBM Classifier's validation classification report is:\n{report_val}")
        print("=" * 100)
        print(f"LightGBM Classifier's testing accuracy is {accuracy_test}")
        print("-" * 70)
        print(f"LightGBM Classifier's testing classification report is:\n{report_test}")

        # Return the trained LightGBM model
        return lgb_model, pred_val, pred_test
    
    # LightGBM + GridSearch for best params
    def LightGBM_GridSearchCV(self, params, cv): 
        """
        Returns:
        best_lgbm: The LightGBM model trained on the best parameters found by GridSearchCV.
        best_params: The best parameters found by GridSearchCV.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.
        """
        # Call a default model
        default = lgb.LGBMClassifier(**self.default_params)

        # GridSearchCV
        grid_search = GridSearchCV(estimator = default, param_grid = params, cv = cv, n_jobs = -1)
        grid_search.fit(self.X_train, self.y_train)

        # Best Parameters
        best_params = grid_search.best_params_

        # Train the model with the best parameters
        best_lgbm = lgb.LGBMClassifier(**best_params, random_state = 42, eval_set = [(self.X_val, self.y_val)])
        best_lgbm.fit(self.X_train, self.y_train)

        # Validation
        pred_val = best_lgbm.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = best_lgbm.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test)

        # Printing the results
        print(f"LightGBM Classifier's validation accuracy (GridSearchCV) is {accuracy_val}")
        print("-" * 70)
        print(f"LightGBM Classifier's validation classification report (GridSearchCV) is:\n{report_val}")
        print("=" * 100)
        print(f"LightGBM Classifier's testing accuracy (GridSearchCV) is {accuracy_test}")
        print("-" * 70)
        print(f"LightGBM Classifier's testing classification report (GridSearchCV) is:\n{report_test}")

        # Return the trained LightGBM model and the best parameters
        return best_lgbm, best_params, pred_val, pred_test
    
    # LightGBM + RandomizedSearchCV for best params
    def LightGBM_RandomizedSearchCV(self, params, cv):
        """
        Returns:
        best_lgbm: The LightGBM model trained on the best parameters found by RandomizedSearchCV.
        best_params: The best parameters found by RandomizedSearchCV.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.
        """

        # Call a default model
        default = lgb.LGBMClassifier(**self.default_params)

        # RandomizedSearchCV
        random_search = RandomizedSearchCV(estimator = default, param_distributions = params, cv = cv, n_jobs = -1)
        random_search.fit(self.X_train, self.y_train)

        # Best Parameters
        best_params = random_search.best_params_

        # Train the model with the best parameters
        best_lgbm = lgb.LGBMClassifier(**best_params, random_state = 42, eval_set = [(self.X_val, self.y_val)])
        best_lgbm.fit(self.X_train, self.y_train)

        # Validation
        pred_val = best_lgbm.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = best_lgbm.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test)

        # Printing the results
        print(f"LightGBM Classifier's validation accuracy (RandomizedSearchCV) is {accuracy_val}")
        print("-" * 70)
        print(f"LightGBM Classifier's validation classification report (RandomizedSearchCV) is:\n{report_val}")
        print("=" * 100)
        print(f"LightGBM Classifier's testing accuracy (RandomizedSearchCV) is {accuracy_test}")
        print("-" * 70)
        print(f"LightGBM Classifier's testing classification report (RandomizedSearchCV) is:\n{report_test}")

        # Return the trained LightGBM model and the best parameters
        return best_lgbm, best_params, pred_val, pred_test
    

    
    def plot_learning_curves(self, model):
        """
        Plot learning curves to visualize the training progress.

        Returns:
        - fig: The Plotly figure showing learning curves.
        """
        # Initialize lists to store training and validation scores
        train_scores = []
        val_scores = []

        # Train the model on increasingly larger subsets of the training data
        for i in range(1, len(self.X_train) + 1):
            subset_X_train = self.X_train[:i]
            subset_y_train = self.y_train[:i]

            # Fit the model
            model.fit(subset_X_train, subset_y_train)

            # Predictions on the training and validation sets
            train_pred = model.predict(subset_X_train)
            val_pred = model.predict(self.X_val)

            # Calculate accuracy on the subsets
            train_accuracy = accuracy_score(subset_y_train, train_pred)
            val_accuracy = accuracy_score(self.y_val, val_pred)

            train_scores.append(train_accuracy)
            val_scores.append(val_accuracy)

        # Create a Plotly figure for the learning curves
        trace1 = go.Scatter(x = list(range(1, len(self.X_train) + 1)), y = train_scores, mode = 'lines', name = 'Train')
        trace2 = go.Scatter(x = list(range(1, len(self.X_train) + 1)), y = val_scores, mode = 'lines', name = 'Validation')

        layout = go.Layout(title = 'Learning Curves', xaxis = dict(title = 'Training Examples'), yaxis = dict(title = 'Accuracy'))
        fig = go.Figure(data = [trace1, trace2], layout = layout)

        return fig
