# Machine Learning Classification Models Repository

This repository is dedicated to various machine learning classification models and related utility classes for efficient and readable code. Each model is encapsulated within a class that follows a common architecture for consistency and ease of use.

## Table of Contents
- [Classes and Functions](#classes-and-functions)
  - [Model Classes](#model-classes)
  - [Methods Class](#methods-class)
  - [TreeOperations Class](#treeoperations-class)
- [List of Supported Models](#list-of-supported-models)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Classes and Functions

### Model Classes (ModelName.py)

1. **(Model)_Classic Function**
   - Create an instance of the class with training, validation, and testing data as parameters.
   - This function trains the model with default parameters, makes predictions, and returns accuracy scores and classification reports for validation and testing data.

2. **(Model)_GridSearchCV Function**
   - Create an instance of the class with training, validation, and testing data as parameters.
   - This function performs hyperparameter tuning using GridSearchCV, trains the model with the best parameters, makes predictions, and prints accuracy scores and classification reports for validation and testing data.

3. **(Model)_RandomizedSearchCV Function**
   - Similar to GridSearchCV function but uses RandomizedSearchCV for hyperparameter tuning.

### Methods Class (Methods.py)

1. **Grid_Search Function**
   - Takes model and parameters as inputs.
   - Finds the best parameters using GridSearchCV without training the model.
   - Returns the best parameters.

2. **Random_Search Function**
   - Similar to Grid_Search but uses RandomizedSearchCV.

3. **rfe_Random_Forest Function**
   - Uses Recursive Feature Elimination (RFE) on a Random Forest model.
   - Returns the RFE model, the self-model trained on RFE-transformed data, and the transformed data.

4. **perform_pca Function**
   - Performs Principal Component Analysis (PCA) on the data.
   - Returns PCA model, self-model trained on PCA-transformed data, and transformed data.

5. **SMOTE_Balancing Function**
   - Performs SMOTE balancing on the target feature.
   - Returns resampled data.

6. **feature_analysis Function**
   - Creates box plots of false positives and false negatives for feature analysis.

### TreeOperations Class (TreeOperations.py)

1. **Visualize_tree Function**
   - Takes features as input and plots the tree for decision tree models.

2. **feature_importance Function**
   - Plots important features as assigned by the tree model.
   - Returns important features.

3. **cost_complexity_prune Function**
   - Prunes Decision Tree models using cost complexity pruning.
   - Returns pruned tree, best ccp_alpha, and index of the least impurity.

## List of Supported Models

- Random Forest
- Balanced Random Forest
- Decision Tree
- XGBoost
- Light Gradient Boosting
- BootstrapAggregator
- Support Vector Classifier (SVC)

## Usage

To use these classes and functions, follow these steps:

1. Import the relevant classes and functions from this repository.
2. Create instances of the model classes, passing training, validation, and testing data.
3. Use the functions provided by these classes for model training, tuning, and analysis.

## Contributing

Contributions to this repository are welcome. If you have improvements, new models, or additional utility functions, please create a pull request.

## License

This repository is licensed under the [MIT License](LICENSE).


