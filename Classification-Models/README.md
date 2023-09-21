# Machine Learning Models

Welcome to the Machine Learning Models section of this project's README. Here, you'll find information about various ML models available for your use. Each model can be used with default parameters or fine-tuned using techniques such as GridSearchCV or RandomizedSearchCV.

## Table of Contents
1. [Support Vector Classifier (SVC)](#support-vector-classifier-svc)
2. [Light Gradient Boosting (LightGBM)](#light-gradient-boosting-lightgbm)
3. [Bootstrap Aggregator (Bagging)](#bootstrap-aggregator-bagging)
4. [Random Forest](#random-forest)
5. [Balanced Random Forest](#balanced-random-forest)
6. [XGBoost](#xgboost)
7. [Decision Tree](#decision-tree)

## 1. Support Vector Classifier (SVC)

The Support Vector Classifier is a classification model that uses support vector machines to classify data.

- **Classic SVC**
  - Implementation: `SupportVectorClassifier.SVC_Classic()`
  - Description: Train and evaluate the classic Support Vector Classifier on your data.
  - Usage Example:
    ```python
    svc_model, pred_val, pred_test = SupportVectorClassifier.SVC_Classic()
    ```

- **SVC with GridSearchCV**
  - Implementation: `SupportVectorClassifier.SVC_GridSearchCV(params, cv)`
  - Description: Fine-tune the Support Vector Classifier using GridSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_svc, best_params, pred_val, pred_test = SupportVectorClassifier.SVC_GridSearchCV(params, cv)
    ```

- **SVC with RandomizedSearchCV**
  - Implementation: `SupportVectorClassifier.SVC_RandomizedSearchCV(params, cv)`
  - Description: Fine-tune the Support Vector Classifier using RandomizedSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_svc, best_params, pred_val, pred_test = SupportVectorClassifier.SVC_RandomizedSearchCV(params, cv)
    ```

## 2. Light Gradient Boosting (LightGBM)

Light Gradient Boosting is a gradient boosting framework that uses tree-based learning algorithms.

- **Classic LightGBM**
  - Implementation: `LightGradient.LightGBM_Classic()`
  - Description: Train and evaluate the classic LightGBM model on your data.
  - Usage Example:
    ```python
    lgbm_model, pred_val, pred_test = LightGradient.LightGBM_Classic()
    ```

- **LightGBM with GridSearchCV**
  - Implementation: `LightGradient.LightGBM_GridSearchCV(params, cv)`
  - Description: Fine-tune the LightGBM model using GridSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_lgbm, best_params, pred_val, pred_test = LightGradient.LightGBM_GridSearchCV(params, cv)
    ```

- **LightGBM with RandomizedSearchCV**
  - Implementation: `LightGradient.LightGBM_RandomizedSearchCV(params, cv)`
  - Description: Fine-tune the LightGBM model using RandomizedSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_lgbm, best_params, pred_val, pred_test = LightGradient.LightGBM_RandomizedSearchCV(params, cv)
    ```

- **Plot Learning Curves**
  - Implementation: `LightGradient.plot_learning_curves(model)`
  - Description: Visualize learning curves to understand the model's training progress.
  - Usage Example:
    ```python
    fig = LightGradient.plot_learning_curves(lgbm_model)
    fig.show()
    ```

## 3. Bootstrap Aggregator (Bagging)

Bootstrap Aggregator, or Bagging, is an ensemble method that combines multiple models to improve performance.

- **Classic Bagging**
  - Implementation: `Bagging.Bagging_Classic()`
  - Description: Train and evaluate the classic Bagging model on your data.
  - Usage Example:
    ```python
    bagging_model, pred_val, pred_test = Bagging.Bagging_Classic()
    ```

- **Bagging with GridSearchCV**
  - Implementation: `Bagging.Bagging_GridSearchCV(params, cv)`
  - Description: Fine-tune the Bagging model using GridSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_bagging, best_params, pred_val, pred_test = Bagging.Bagging_GridSearchCV(params, cv)
    ```

- **Bagging with RandomizedSearchCV**
  - Implementation: `Bagging.Bagging_RandomizedSearchCV(params, cv)`
  - Description: Fine-tune the Bagging model using RandomizedSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_bagging, best_params, pred_val, pred_test = Bagging.Bagging_RandomizedSearchCV(params, cv)
    ```

## 4. Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and control overfitting.

- **Classic Random Forest**
  - Implementation: `RandomForest.Random_Forest_Classic()`
  - Description: Train and evaluate the classic Random Forest model on your data.
  - Usage Example:
    ```python
    rf_model, pred_val, pred_test = RandomForest.Random_Forest_Classic()
    ```

- **Random Forest + GridSearch for best params**
  - Implementation: `RandomForest.Random_Forest_GridSearchCV(params, cv)`
  - Description: Fine-tune the Random Forest model using GridSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_rf, best_params, pred_val, pred_test = RandomForest.Random_Forest_GridSearchCV(params, cv)
    ```

- **Random Forest + RandomizedSearch for best params**
  - Implementation: `RandomForest.Random_Forest_RandomizedSearchCV(params, cv)`
  - Description: Fine-tune the Random Forest model using RandomizedSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_rf, best_params, pred_val, pred_test = RandomForest.Random_Forest_RandomizedSearchCV(params, cv)
    ```

## 5. Balanced Random Forest

Balanced Random Forest is a variation of Random Forest designed for imbalanced datasets.

- **Classic Balanced Random Forest**
  - Implementation: `BalancedRandomForestModel.BRFC_Classic()`
  - Description: Train and evaluate the classic Balanced Random Forest model on your data.
  - Usage Example:
    ```python
    brf_model, pred_val, pred_test = BalancedRandomForestModel.BRFC_Classic()
    ```

- **Balanced Random Forest + GridSearch for best params**
  - Implementation: `BalancedRandomForestModel.BRFC_GridSearchCV(params, cv)`
  - Description: Fine-tune the Balanced Random Forest model using GridSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_brf, best_params, pred_val, pred_test = BalancedRandomForestModel.BRFC_GridSearchCV(params, cv)
    ```

- **Balanced Random Forest + RandomizedSearch for best params**
  - Implementation: `BalancedRandomForestModel.BRFC_RandomizedSearchCV(params, cv)`
  - Description: Fine-tune the Balanced Random Forest model using RandomizedSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_brf, best_params, pred_val, pred_test = BalancedRandomForestModel.BRFC_RandomizedSearchCV(params, cv)
    ```

## 6. XGBoost

XGBoost is a gradient boosting algorithm known for its speed and performance.

- **Classic XGBoost**
  - Implementation: `XGBoostClassifier.XGBoost_Classic()`
  - Description: Train and evaluate the classic XGBoost model on your data.
  - Usage Example:
    ```python
    xgb_model, pred_val, pred_test = XGBoostClassifier.XGBoost_Classic()
    ```

- **XGBoost + GridSearch for best params**
  - Implementation: `XGBoostClassifier.XGBoost_GridSearchCV(params, cv)`
  - Description: Fine-tune the XGBoost model using GridSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_xgb, best_params, pred_val, pred_test = XGBoostClassifier.XGBoost_GridSearchCV(params, cv)
    ```

- **XGBoost + RandomizedSearch for best params**
  - Implementation: `XGBoostClassifier.XGBoost_RandomizedSearchCV(params, cv)`
  - Description: Fine-tune the XGBoost model using RandomizedSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_xgb, best_params, pred_val, pred_test = XGBoostClassifier.XGBoost_RandomizedSearchCV(params, cv)
    ```

## 7. Decision Tree

Decision Trees are simple yet powerful models used for classification tasks.

- **Classic Decision Tree**
  - Implementation: `DecisionTree.Decision_Tree_Classic()`
  - Description: Train and evaluate the classic Decision Tree model on your data.
  - Usage Example:
    ```python
    dt_model, pred_val, pred_test = DecisionTree.Decision_Tree_Classic()
    ```

- **Decision Tree + GridSearch for best params**
  - Implementation: `DecisionTree.Decision_Tree_GridSearchCV(params, cv)`
  - Description: Fine-tune the Decision Tree model using GridSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_dt, best_params, pred_val, pred_test = DecisionTree.Decision_Tree_GridSearchCV(params, cv)
    ```

- **Decision Tree + RandomizedSearch for best params**
  - Implementation: `DecisionTree.Decision_Tree_RandomizedSearchCV(params, cv)`
  - Description: Fine-tune the Decision Tree model using RandomizedSearchCV to find the best hyperparameters.
  - Usage Example:
    ```python
    best_dt, best_params, pred_val, pred_test = DecisionTree.Decision_Tree_RandomizedSearchCV(params, cv)
    ```

Feel free to explore these models, customize their parameters, and choose the one that best fits your machine learning task. Happy modeling!

