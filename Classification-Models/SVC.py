from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report


class SupportVectorClassifier:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # The model to use search on
        self.svc = SVC(random_state = 42)
        # Training
        self.X_train = X_train
        self.y_train = y_train

        # Validation
        self.X_val = X_val
        self.y_val = y_val

        # Testing
        self.X_test = X_test
        self.y_test = y_test

    # Support Vector Classifier with default params
    def SVC_Classic(self):

        """
        Returns:
        self.svc: The classic Support Vector Classifier model trained on the passed data.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.
        """

        # Train model
        self.svc.fit(self.X_train, self.y_train)

        # Validation
        pred_val = self.svc.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = self.svc.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test)

        # Printing the results
        print(f"SVC's validation accuracy is {accuracy_val}")
        print("-" * 70)
        print(f"SVC's validation classification report is:\n{report_val}")
        print("=" * 100)
        print(f"SVC's testing accuracy is {accuracy_test}")
        print("-" * 70)
        print(f"SVC's testing classification report is:\n{report_test}")

        # Return the trained SVC model
        return self.svc, pred_val, pred_test

    # Support Vector Classifier + GridSearch for best params
    def SVC_GridSearchCV(self, params, cv):

        """
        Returns:
        best_svc: The Support Vector Classifier model trained on the best parameters found by GridSearchCV.
        best_params: The best parameters found by GridSearchCV.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.
        """
        # GridSearchCV
        grid_search = GridSearchCV(self.svc, params, cv = cv, n_jobs = -1)
        grid_search.fit(self.X_train, self.y_train)

        # Best Parameters
        best_params = grid_search.best_params_

        # Train the model with the best parameters
        best_svc = SVC(**best_params, random_state = 42)
        best_svc.fit(self.X_train, self.y_train)

        # Validation
        pred_val = best_svc.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = best_svc.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test)

        # Printing the results
        print(f"SVC's validation accuracy (GridSearchCV) is {accuracy_val}")
        print("-" * 70)
        print(f"SVC's validation classification report (GridSearchCV) is:\n{report_val}")
        print("=" * 100)
        print(f"SVC's testing accuracy (GridSearchCV) is {accuracy_test}")
        print("-" * 70)
        print(f"SVC's testing classification report (GridSearchCV) is:\n{report_test}")

        # Return the trained SVC model and the best parameters
        return best_svc, best_params, pred_val, pred_test

    # Support Vector Classifier + RandomizedSearchCV for best params
    def SVC_RandomizedSearchCV(self, params, cv):

        """
        Returns:
        best_svc: The Support Vector Classifier model trained on the best parameters found by RandomizedSearchCV.
        best_params: The best parameters found by RandomizedSearchCV.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.

        """
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(self.svc, params, cv = cv, n_jobs = -1)
        random_search.fit(self.X_train, self.y_train)

        # Best Parameters
        best_params = random_search.best_params_

        # Train the model with the best parameters
        best_svc = SVC(**best_params, random_state = 42)
        best_svc.fit(self.X_train, self.y_train)

        # Validation
        pred_val = best_svc.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = best_svc.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test)

        # Printing the results
        print(f"SVC's validation accuracy (RandomizedSearchCV) is {accuracy_val}")
        print("-" * 70)
        print(f"SVC's validation classification report (RandomizedSearchCV) is:\n{report_val}")
        print("=" * 100)
        print(f"SVC's testing accuracy (RandomizedSearchCV) is {accuracy_test}")
        print("-" * 70)
        print(f"SVC's testing classification report (RandomizedSearchCV) is:\n{report_test}")

        # Return the trained SVC model and the best parameters
        return best_svc, best_params, pred_val, pred_test
