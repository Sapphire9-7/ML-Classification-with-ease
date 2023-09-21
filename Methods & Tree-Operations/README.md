# Machine Learning Methods

This section provides information about two Python classes: `Methods` and `TreeOperations`. These classes offer various machine learning operations and tree-related functions. You can use these classes to streamline your machine learning workflow.

## 1. Methods Class

The `Methods` class offers several useful methods for hyperparameter tuning, feature selection, and more. It includes the following methods:

- **Grid_Search**
  - Description: Perform hyperparameter tuning using GridSearchCV.
  - Usage Example:
    ```python
    best_params = Methods(model).Grid_Search(parameters, X_train, y_train, cv)
    ```

- **Random_Search**
  - Description: Perform hyperparameter tuning using RandomizedSearchCV.
  - Usage Example:
    ```python
    best_params = Methods(model).Random_Search(params, iterations, cv, random_state, X_train, y_train)
    ```

- **rfe_Random_Forest**
  - Description: Perform feature selection using Recursive Feature Elimination (RFE) with a Random Forest model.
  - Usage Example:
    ```python
    rfe, model_rfe_trained, X_train_rfe, X_val_selected, X_test_selected = Methods(model).rfe_Random_Forest(X_train, y_train, X_val, y_val, X_test, y_test, num_features_to_keep)
    ```

- **perform_Pca**
  - Description: Perform Principal Component Analysis (PCA) and train a model on the transformed data.
  - Usage Example:
    ```python
    pca_model, model_pca_trained, X_train_pca, X_val_pca, X_test_pca = Methods(model).perform_Pca(X_train, y_train, X_val, y_val, X_test, y_test, num_components)
    ```

- **SMOTE_Balancing**
  - Description: Apply Synthetic Minority Over-sampling Technique (SMOTE) for class balancing.
  - Usage Example:
    ```python
    X_train_resampled, y_train_resampled = Methods(model).SMOTE_Balancing(X_train, y_train)
    ```

- **feature_analysis**
  - Description: Analyze features that contribute to false positives and false negatives.
  - Usage Example:
    ```python
    Methods(model).feature_analysis(X, y_true, y_pred, title)
    ```

## 2. TreeOperations Class

The `TreeOperations` class focuses on operations related to decision tree models and includes the following methods:

- **Visualize_tree**
  - Description: Visualize the decision tree model.
  - Usage Example:
    ```python
    TreeOperations(model, X_train, y_train, X_val, y_val, X_test, y_test).Visualize_tree(X)
    ```

- **feature_importance**
  - Description: Find and visualize feature importance using the decision tree model.
  - Usage Example:
    ```python
    feature_importance = TreeOperations(model, X_train, y_train, X_val, y_val, X_test, y_test).feature_importance(X)
    ```

- **cost_complexity_prune**
  - Description: Prune the decision tree model using cost-complexity pruning.
  - Usage Example:
    ```python
    ds_pruned, lowest_impurity_alpha, min_impurity_idx = TreeOperations(model, X_train, y_train, X_val, y_val, X_test, y_test).cost_complexity_prune()
    ```

Feel free to use these classes and methods in your machine learning project to streamline your workflow, improve model performance, and gain insights into your data and models.
