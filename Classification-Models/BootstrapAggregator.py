from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

class Bagging:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, base_estimator):
        # The model to use search on
        self.bagging = BaggingClassifier(base_estimator = base_estimator, random_state = 42)

        # Training
        self.X_train = X_train
        self.y_train = y_train

        # Validation
        self.X_val = X_val
        self.y_val = y_val

        # Testing
        self.X_test = X_test
        self.y_test = y_test

    # Bagging with default params
    def Bagging_Classic(self):
        """
        Returns:
        self.bagging: The classic Bagging model trained on the passed data.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.

        """
        # Train model
        self.bagging.fit(self.X_train, self.y_train)

        # Validation
        pred_val = self.bagging.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = self.bagging.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test)

        # Printing the results
        print(f"Bagging Classifier's validation accuracy is {accuracy_val}")
        print("-" * 70)
        print(f"Bagging Classifier's validation classification report is: \n {report_val}")
        print("=" * 100)
        print(f"Bagging Classifier's testing accuracy is {accuracy_test}")
        print("-" * 70)
        print(f"Bagging Classifier's testing classification report is: \n {report_test}")

        # Return the trained Bagging model
        return self.bagging, pred_val, pred_test

    # Bagging + GridSearch for best params
    def Bagging_GridSearchCV(self, params, cv):
        """
        Returns:
        best_bagging: The Bagging model trained on the best parameters found by the GridSearchCV.
        best_params: The best parameters found by the GridSearchCV.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.

        """
        # GridSearchCV
        grid_search = GridSearchCV(self.bagging, params, cv = cv, n_jobs = -1)
        grid_search.fit(self.X_train, self.y_train)

        # Best Parameters
        best_params = grid_search.best_params_

        # Train the model with the best parameters
        best_bagging = BaggingClassifier(base_estimator = self.bagging.base_estimator, **best_params)
        best_bagging.fit(self.X_train, self.y_train)

        # Validation
        pred_val = best_bagging.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = best_bagging.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test)

        # Printing the results
        print(f"Bagging Classifier's validation accuracy (GridSearchCV) is {accuracy_val}")
        print("-" * 70)
        print(f"Bagging Classifier's validation classification report (GridSearchCV) is: \n {report_val}")
        print("=" * 100)
        print(f"Bagging Classifier's testing accuracy (GridSearchCV) is {accuracy_test}")
        print("-" * 70)
        print(f"Bagging Classifier's testing classification report (GridSearchCV) is: \n {report_test}")

        # Return the trained Bagging model and the best parameters
        return best_bagging, best_params, pred_val, pred_test

    # Bagging + RandomizedSearchCV for best params
    def Bagging_RandomizedSearchCV(self, params, cv):
        """
        Returns:
        best_bagging: The Bagging model trained on the best parameters found by RandomizedSearchCV.
        best_params: The best parameters found by RandomizedSearchCV.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.

        """
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(self.bagging, params, cv = cv, n_jobs = -1)
        random_search.fit(self.X_train, self.y_train)

        # Best Parameters
        best_params = random_search.best_params_

        # Train the model with the best parameters
        best_bagging = BaggingClassifier(base_estimator = self.bagging.base_estimator, **best_params)
        best_bagging.fit(self.X_train, self.y_train)

        # Validation
        pred_val = best_bagging.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = best_bagging.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test)

        # Printing the results
        print(f"Bagging Classifier's validation accuracy (RandomizedSearchCV) is {accuracy_val}")
        print("-" * 70)
        print(f"Bagging Classifier's validation classification report (RandomizedSearchCV) is:\n{report_val}")
        print("=" * 100)
        print(f"Bagging Classifier's testing accuracy (RandomizedSearchCV) is {accuracy_test}")
        print("-" * 70)
        print(f"Bagging Classifier's testing classification report (RandomizedSearchCV) is:\n{report_test}")

        # Return the trained Bagging model and the best parameters
        return best_bagging, best_params, pred_val, pred_test
