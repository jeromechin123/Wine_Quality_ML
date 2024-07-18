# Wine_Quality_ML
A study using different models to predict wine quality using different classifiers. 


# Summary of Findings
Based on the dataset and the classification models' performance, here's a detailed summary of the findings:

The classifier models are ran through individually to get the accuracy, classification report and feature importance for each variable.

Having tuned the parameters for xgboost, the accuracy was found to be the highest amongst all models and thus was use to explain the findings. 

A correlation matrix was produced to see whether the features correlate positively or negatively. 

# Dataset Information
- Dataset Name: Wine Quality
- Source: UCI Machine Learning Repository
- Description: The dataset contains physicochemical tests results for red and white wine samples from the north of Portugal. The goal is to model wine quality based on these tests.
- Number of Instances: 4,898
- Number of Features: 11 (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
- Target Variable: Quality (score between 0 and 10)
- https://archive.ics.uci.edu/dataset/186/wine+quality

# Model Performance
The following models were evaluated on the dataset, and their performance metrics were recorded:

## 1. Decision Tree Classifier

- Accuracy: 58.46%
- Confusion Matrix: Shows misclassifications across different classes.
- Precision, Recall, F1-Score: Varying performance with a tendency to overfit due to the tree's depth.

## 2. Random Forest Classifier

- Accuracy: 67.26%
- Confusion Matrix: Better performance than the decision tree, but some misclassifications persist.
- Precision, Recall, F1-Score: Improved compared to the decision tree due to the ensemble approach.

## 3. Gradient Boosting Classifier

- Accuracy: 59.57%
- Confusion Matrix: Shows similar trends as the decision tree but with better handling of some classes.
- Precision, Recall, F1-Score: Indicates potential underfitting or need for hyperparameter tuning.

## 4. XGBoost Classifier

- Assumption that 

- Accuracy: 68.08%
- Confusion Matrix: Similar trends as gradient boosting with better handling of certain classes.
- Precision, Recall, F1-Score: Suggests balanced performance but still requires a lot fine-tuning for optimal results.
- The classifier was initialised with
  ```
  xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',  # for multiclass classification
    num_class=10,               # number of classes in the target variable
    max_depth=8,                # maximum depth of the trees 
    learning_rate=0.4,          # learning rate
    n_estimators=500,           # number of trees (boosting rounds)
    random_state=20
  )
    ```


# Key Metrics Interpretation for the Classification Report
- Precision: Indicates the proportion of true positive predictions among the total predicted positives. Higher precision for certain classes means the model is more accurate when it predicts those classes.
- Recall (Sensitivity): Represents the proportion of true positive predictions among the actual positives. Higher recall indicates the model's ability to capture most of the positive instances.
- F1 Score: The harmonic mean of precision and recall. It provides a balance between the two, especially useful for imbalanced datasets.
- Support: The number of actual occurrences of each class in the dataset, providing context for the other metrics.


# Feature Importance Analysis with XGBoost
The feature importance analysis with XGBoost revealed that 'alcohol' is the most significant predictor of wine quality, followed by 'volatile_acidity' and 'sulphates'. The feature importance chart indicates the following:

- Alcohol: The most important feature, significantly impacting wine quality. A positive correlation suggest wines are higher in quality when alcohol% is higher. 
- Volatile Acidity: The second most important feature. Wine quality is lower when there are more Volatile Acids
- Sulphates: Also plays a crucial role in determining wine quality.
- Other features such as 'residual_sugar', 'free_sulfur_dioxide', and 'density' also contribute but to a lesser extent.

![XGboost_feature_importance](https://github.com/user-attachments/assets/1c85ad3d-dfc4-4193-8a07-df26b48631bd)

![corr](https://github.com/user-attachments/assets/89ee3879-31d6-4275-9285-44a0e69fce15)

# Conclusion

- The classification models can be improved to fit better with tuning and depth limiting.
- By combining results from multiple models such as blending to create an ensemble of models can potentially improve the results.
- The data is skewed heavily to a quality score of 5 which suggest that there are outliers in low and high quality wines. The model prediction performance for higher and lower quality wine is thus very inaccurate.
  
