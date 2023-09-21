from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report


class RandomForest:
    def __init__(self, X_train,y_train, X_val, y_val, X_test, y_test):
        # The model to use search on
        self.rf = RandomForestClassifier(random_state = 42)
        #Training
        self.X_train = X_train
        self.y_train = y_train

        # Validation
        self.X_val = X_val
        self.y_val = y_val

        # Testing
        self.X_test = X_test
        self.y_test = y_test
    
    # Random Forest with default params
    def Random_Forest_Classic(self):

        """
        Returns:
        self.rf: The classic Random Forest model trained on the passed data.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.

        """

        # Train model
        self.rf.fit(self.X_train, self.y_train)
        
        # Validation
        pred_val = self.rf.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = self.rf.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test) 

        # Printing the results
        print(f"Random Forest Classifier's validation accuracy is {accuracy_val}")
        print("-"*70)
        print(f"Random Forest Classifier's validation classification report is: \n {report_val}")
        print("="*100)
        print(f"Random Forest Classifier's testing accuracy is {accuracy_test}")
        print("-"*70)
        print(f"Random Forest Classifier's testing classification report is: \n {report_test}")

        # Return the trained Random Forest model
        return self.rf, pred_val, pred_test
    

    # Random Forest + GridSearch for best params
    def Random_Forest_GridSearchCV(self, params, cv): 
        
        """
        Returns:
        best_rf: The Random Forest model trained on the best parameters found by the GridSearchCV.
        best_params: The best parameters found by the GridSearchCV.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.
        
        """
        # GridSearchCV
        grid_search = GridSearchCV(self.rf, params, cv = cv, n_jobs = -1)
        grid_search.fit(self.X_train, self.y_train)

        # Best Parameters
        best_params = grid_search.best_params_

        # Train the model with the best parameters
        best_rf = RandomForestClassifier(**best_params)
        best_rf.fit(self.X_train, self.y_train)

        # Validation
        pred_val = best_rf.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = best_rf.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test) 

        # Printing the results
        print(f"Random Forest Classifier's validation accuracy (GridSearchCV) is {accuracy_val}")
        print("-"*70)
        print(f"Random Forest Classifier's validation classification report (GridSearchCV) is: \n {report_val}")
        print("="*100)
        print(f"Random Forest Classifier's testing accuracy (GridSearchCV) is {accuracy_test}")
        print("-"*70)
        print(f"Random Forest Classifier's testing classification report (GridSearchCV) is: \n {report_test}")

        # Return the trained Random Forest model and the best parameters
        return best_rf, best_params, pred_val, pred_test
    

    # Random Forest + RandomSearchCV for best params
    def Random_Forest_RandomizedSearchCV(self, params, cv):

        """
        Returns:
        best_rf: The Random Forest model trained on the best parameters found by RandomSearchCV.
        best_params: The best parameters found by RandomSearchCV.
        pred_val: Model prediction on validation data.
        pred_test: Model prediction on testing data.
        
        """
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(self.rf, params, cv = cv, n_jobs = -1)
        random_search.fit(self.X_train, self.y_train)

        # Best Parameters
        best_params = random_search.best_params_

        # Train the model with the best parameters
        best_rf = RandomForestClassifier(**best_params)
        best_rf.fit(self.X_train, self.y_train)

        # Validation
        pred_val = best_rf.predict(self.X_val)
        # Evaluating validation
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = best_rf.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test)

        # Printing the results
        print(f"Random Forest Classifier's validation accuracy (RandomizedSearchCV) is {accuracy_val}")
        print("-" * 70)
        print(f"Random Forest Classifier's validation classification report (RandomizedSearchCV) is:\n{report_val}")
        print("=" * 100)
        print(f"Random Forest Classifier's testing accuracy (RandomizedSearchCV) is {accuracy_test}")
        print("-" * 70)
        print(f"Random Forest Classifier's testing classification report (RandomizedSearchCV) is:\n{report_test}")

        # Return the trained Random Forest model and the best parameters
        return best_rf, best_params, pred_val, pred_test



        


