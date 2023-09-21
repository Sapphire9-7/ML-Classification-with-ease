from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree


# pass the trained model
class TreeOperations:


    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        # The model to perform tree operations on
        self.model = model

        # Training
        self.X_train = X_train
        self.y_train = y_train

        # Validation
        self.X_val = X_val
        self.y_val = y_val

        # Testing
        self.X_test = X_test
        self.y_test = y_test
    
    # Visualize the tree.
    def Visualize_tree(self, X):
        # Figure size
        plt.figure(figsize = (20, 20))
        # Plot the tree
        tree.plot_tree(self.model, filled = True, feature_names = list(X.columns), 
                       class_names = ['No', 'Yes'], rounded = True)
    
    # Find the model's important features and plot them, return the important features.
    def feature_importance(self, X):
        """
        Returns:
        importance: The importanct features assigned by the tree model.

        """
        # Get the important features
        importance = self.model.feature_importances_
        feature_importance = pd.Series(importance, index = X.columns)
        # Plot the important features
        plt.figure(figsize = (15, 10))
        feature_importance.plot(kind = 'barh')
        plt.ylabel('Features')

        return feature_importance
    

    # Find ccp_alpha for the model, plot, find lowest ccp_alpha, re-train model using designated ccp_alpha. (pre-pruning technique)
    def cost_complexity_prune(self):

        """
        Returns:
        ds_pruned: The pruned tree model.
        lowest_impurity_alpha: The ccp_alpha with the lowest impurity (used in pruning)
        min_impurity_idx: Index of the lowest impurity

        """
         
        # Initialize tree pruning
        path = self.model.cost_complexity_pruning_path(self.X_train, self.y_train)
        # Find the ccp_alphas and impurities
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        # Find the index of the minimum impurity
        min_impurity_idx = np.argmin(impurities)
        # Get the corresponding ccp_alpha
        lowest_impurity_alpha = ccp_alphas[min_impurity_idx]

        # Plot alphas against impurities
        plt.figure(figsize = (10, 6))
        plt.plot(ccp_alphas[:-1], impurities[:-1], marker = 'o', drawstyle = 'steps-post')
        plt.xlabel('effective alpha')
        plt.ylabel('total impurity of leaves')
        plt.title('Total Impurity vs effective alpha for training set')

        # Print the alpha with the lowest impurity
        print(f"The ccp alpha with the lowest impurity is {lowest_impurity_alpha}")

        # Re-train model with designated ccp_alpha
        ds_pruned = DecisionTreeClassifier(random_state = 42, ccp_alpha = lowest_impurity_alpha)
        ds_pruned.fit(self.X_train, self.y_train)

        # Evaluate
        # Validation
        pred_val = ds_pruned.predict(self.X_val)
        # Accuracy
        accuracy_val = accuracy_score(self.y_val, pred_val)
        # Classification report
        report_val = classification_report(self.y_val, pred_val)

        # Testing
        pred_test = ds_pruned.predict(self.X_test)
        # Evaluating testing
        # Accuracy
        accuracy_test = accuracy_score(self.y_test, pred_test)
        # Classification report
        report_test = classification_report(self.y_test, pred_test) 

        # Printing the results
        print(f"Decision Tree Classifier's validation accuracy is {accuracy_val}")
        print("-"*70)
        print(f"Decision Tree Classifier's validation classification report is: \n {report_val}")
        print("="*100)
        print(f"Decision Tree Classifier's testing accuracy is {accuracy_test}")
        print("-"*70)
        print(f"Decision Tree Classifier's testing classification report is: \n {report_test}")

        # Return the pruned model, the alpha with lowest impurity, and index of lowest impurity.
        return ds_pruned, lowest_impurity_alpha, min_impurity_idx
    







