from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class Methods:
    def __init__(self, model):
        # The model to use search on
        self.model = model

    # GridSearchCV

    # Pass the model, send the parameters to loop through, send the training data, send the CVs
    def Grid_Search(self, parameters, X_train, y_train, cv):
        # Initialize the search model
        cv = GridSearchCV(self.model, parameters, cv = cv, n_jobs = -1)
        # Train the model
        cv.fit(X_train, y_train.values.ravel())
        
        best_params = cv.best_params_
        means = cv.cv_results_['mean_test_score']
        stds = cv.cv_results_['std_test_score']
        
        # Print the process
        print("BEST PARAMETERS: {}\n".format(best_params))
        for mean, std, params in zip(means, stds, cv.cv_results_['params']):
            print('{} (+/-{}) for {}'.format(round(mean, 3), round(std*2, 3), params))

        # Return the best parameters
        return best_params

    # RandomSearchCV

    # Pass the model, parameters to loop through, num of iterations, CVs, and random_state.
    def Random_Search(self, params, iterations, cv, random_state, X_train, y_train):
        # Initialize the search model
        rs = RandomizedSearchCV(estimator = self.model, param_distributions = params, n_iter = iterations,
                                 cv = cv, n_jobs = -1, random_state = random_state)
        # Train the model
        rs.fit(X_train, y_train)

        means = rs.cv_results_['mean_test_score']
        stds = rs.cv_results_['std_test_score']

        # Print the process
        print("BEST PARAMETERS: {}\n".format(rs.best_params_))
        for mean, std, params in zip(means, stds, rs.cv_results_['params']):
            print('{} (+/-{}) for {}'.format(round(mean, 3), round(std*2, 3), params))
        
        # Return the best parameters
        return rs.best_params_



    # Feature Selection (Finding selected features using Random Forest Classifier) + training passed model on selected features.

    def rfe_Random_Forest(self, X_train, y_train, X_val, y_val, X_test, y_test, num_features_to_keep):
        """
        Returns:
        model_rfe_trained: The passed model trained on transformed selected data.
        rfe: Trained RFE model.
        X_train_selected: Transformed training data.
        X_val_selected: Transformed validation data.
        X_test_selected: Transformed test data.

        """
        # Initialize a random forest model to use in rfe
        rf = RandomForestClassifier(random_state = 42)

        # Number of features to keep
        num_features = num_features_to_keep

        # Initialize RFE
        rfe = RFE(estimator = rf, n_features_to_select = num_features)

        # Fit RFE on training data
        X_train_rfe = rfe.fit_transform(X_train, y_train)

        # Print the process
        # For loop for RFE to print the important features for each round
        for i in range(num_features_to_keep, 0, -1):
            print(f"Round {i}: Selected features - {', '.join(X_train.columns[rfe.support_])}")
            if i > 1:
                eliminated_feature = X_train.columns[np.where(rfe.ranking_ == i)[0][0]]
                print(f"Eliminated feature: {eliminated_feature}")
            print()

        # Transform the feature data
        X_val_selected = rfe.transform(X_val)
        X_test_selected = rfe.transform(X_test)

        # Calculate the permutation importance
        perm_importance = permutation_importance(rfe.estimator_, X_train_rfe, y_train, n_repeats = 30, random_state = 42)
        feature_importances = perm_importance.importances_mean

        # Train the passed model on the selected features
        model_rfe_trained = self.model.fit(X_train_rfe, y_train)

        # Evaluation
        # Validation
        # Accuracy
        y_val_pred = model_rfe_trained.predict(X_val_selected)
        accuracy_val = accuracy_score(y_val, y_val_pred)
        # Classification Report
        report_val = classification_report(y_val, y_val_pred)
        # Print results
        print(f"Validation Accuracy with Selected Features: {accuracy_val}")
        print("-" * 70)
        print(f"Validation classification report with Selected Features: {report_val}")
        
        print("=" * 100)

        # Testing
        # Accuracy
        y_test_pred = model_rfe_trained.predict(X_test_selected)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        # Classification Report
        report_test = classification_report(y_test, y_test_pred)
        # Print results
        print(f"Test Accuracy with Selected Features: {accuracy_test}")
        print("-" * 70)
        print(f"Test classification report with Selected Features: {report_test}")


        # Plot the permutation importance
        sorted_idx = np.argsort(feature_importances)

        plt.figure(figsize = (10, 6))
        plt.barh(range(num_features_to_keep), feature_importances[sorted_idx], align = "center")
        plt.yticks(range(num_features_to_keep), np.array(X_train.columns)[rfe.support_][sorted_idx])
        plt.xlabel("Permutation Importance")
        plt.ylabel("Selected Features")
        plt.title("Permutation Importance Plot")
        # Show
        plt.show()

        # Return the trained model with the transformed data
        return rfe, model_rfe_trained, X_train_rfe, X_val_selected, X_test_selected
    

    # Principle Component Analysis (PCA) + train model on transformed data and evaluate it.

    def perform_Pca(self,X_train, y_train, X_val, y_val, X_test, y_test, num_components, plot_variance = True):
        
        """
        Returns:
        model_pca_trained: The passed model trained on transformed extracted data.
        pca_model: Trained PCA model.
        X_train_pca: Transformed training data.
        X_val_pca: Transformed validation data.
        X_test_pca: Transformed test data.
        
        """
        # Initialize the PCA model
        pca_model = PCA(n_components = num_components)

        # Fit on training data and transform validation and testing data
        X_train_pca = pca_model.fit_transform(X_train)
        X_val_pca = pca_model.transform(X_val)
        X_test_pca = pca_model.transform(X_test)

        # Plot explained variance ratio
        if plot_variance:
            explained_variance = pca_model.explained_variance_ratio_
            cumulative_variance = explained_variance.cumsum()
            
            plt.figure(figsize = (8, 4))
            plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker = 'o', linestyle = '-', color = 'b')
            plt.title('Cumulative Explained Variance')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.grid(True)
            plt.show()
        
        # Train on the passed model
        model_pca_trained = self.model.fit(X_train_pca, y_train)

        # Evaluation
        # Validation
        # Accuracy
        y_val_pred = model_pca_trained.predict(X_val_pca)
        accuracy_val = accuracy_score(y_val, y_val_pred)
        # Classification Report
        report_val = classification_report(y_val, y_val_pred)
        # Print results
        print(f"Validation Accuracy with Extracted Features: {accuracy_val}")
        print("-" * 70)
        print(f"Validation classification report with Extracted Features: {report_val}")
        
        print("=" * 100)

        # Testing
        # Accuracy
        y_test_pred = model_pca_trained.predict(X_test_pca)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        # Classification Report
        report_test = classification_report(y_test, y_test_pred)
        # Print results
        print(f"Test Accuracy with Extracted Features: {accuracy_test}")
        print("-" * 70)
        print(f"Test classification report with Extracted Features: {report_test}")

        # Return the model and transformed data
        return pca_model, model_pca_trained, X_train_pca, X_val_pca, X_test_pca
    

    # SMOTE Balancing
    def SMOTE_Balancing(self, X_train, y_train):
        """
        Returns:
        X_train_resampled: The resampled training data.
        y_train_resampled: The resampled training labels.
        """
        # Initialize SMOTE
        smote = SMOTE(random_state = 42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Class distribution before and after SMOTE
        original_class_distribution = dict(zip(*np.unique(y_train, return_counts = True)))
        resampled_class_distribution = dict(zip(*np.unique(y_train_resampled, return_counts = True)))
        
        print("Class Distribution Before SMOTE:")
        print(original_class_distribution)
        
        print("\nClass Distribution After SMOTE:")
        print(resampled_class_distribution)
        
        return X_train_resampled, y_train_resampled
    


    # Function for feature analysis
    def feature_analysis(self, X, y_true, y_pred, title):
        
        # Convert boolean arrays to indices
        false_positives_indices = np.where((y_true == 0) & (y_pred == 1))[0]
        false_negatives_indices = np.where((y_true == 1) & (y_pred == 0))[0]

        # Select corresponding rows from X based on the indices
        false_positives = X.iloc[false_positives_indices]
        false_negatives = X.iloc[false_negatives_indices]

        # Compute statistics for false positives and false negatives
        fp_mean = false_positives.mean()
        fp_median = false_positives.median()
        fp_std = false_positives.std()

        fn_mean = false_negatives.mean()
        fn_median = false_negatives.median()
        fn_std = false_negatives.std()

        # Create visualizations
        num_features = X.shape[1]
        feature_names = X.columns.tolist()

        fig, axes = plt.subplots(num_features, 2, figsize=(12, 3*num_features))

        for i, feature in enumerate(feature_names):
            axes[i, 0].boxplot([false_positives[feature], false_negatives[feature]], labels=["False Positives", "False Negatives"])
            axes[i, 0].set_xlabel("Prediction")
            axes[i, 0].set_ylabel(feature)
            axes[i, 0].set_title(f"{title}: Comparison of {feature}")
                
            axes[i, 1].hist([false_positives[feature], false_negatives[feature]], bins=20, alpha=0.5, label=["False Positives", "False Negatives"])
            axes[i, 1].set_xlabel(feature)
            axes[i, 1].set_ylabel("Frequency")
            axes[i, 1].set_title(f"{title}: Distribution of {feature}")
            axes[i, 1].legend()

        plt.tight_layout()
        plt.show()
    

    










        
        

        
        
