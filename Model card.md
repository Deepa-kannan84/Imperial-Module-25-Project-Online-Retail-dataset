# Model Card for Online Retail II - Dataset

## 1. KNN
### Model Description
This model predicts the quantity demand of products in an online retail dataset using historical transaction data.

### Input
The inputs include numerical and categorical features: product code (StockCode), price, total price, country, and date features extracted from InvoiceDate such as day of the week, month, and year. The target is the quantity sold.

### Output
The model outputs a continuous numerical value representing the predicted quantity demand for a product on a given day.

### Model Architecture
The model is a K-Nearest Neighbors regressor. Hyperparameters such as number of neighbors, weight type, and distance metric were tuned using randomized search with cross-validation. Categorical variables were one-hot encoded and numerical features were scaled using RobustScaler to handle skewness and outliers.

### Performance
The model was trained and evaluated on a time-ordered train-test split without shuffling. Performance was measured using R² score (47% of the variation in quantity can be explained by the features used), mean squared error (MSE), and mean absolute percentage error (MAPE). The accuracy calculated as 100 minus MAPE was 74.62%. The results demonstrate how well the model predicts unseen quantity demand data.

### Limitations
KNN can be computationally expensive with large datasets. It does not capture long-term trends or seasonality well and depends heavily on the quality of scaling. Sudden changes in demand may not be predicted accurately since KNN relies on similarity to past data points.

### Trade-offs
KNN offers simplicity and interpretability but may underperform compared to more complex models like ARIMA in capturing temporal trends. It is well-suited for local pattern recognition but less effective for forecasting longer-term or seasonal shifts.


## 2. ARIMA
### Model Description
I used an ARIMA model to predict daily total sales from Online Retail II. Input is daily summed sales by date. The model forecasts sales for 15 days ahead with confidence intervals. The data was stationary, so no differencing was needed.

### Input
Daily total sales grouped by date.

### Output
Predicted sales for the next 15 days with 95% confidence intervals.

### Model Architecture
ARIMA with p=3, d=0, q=3, chosen by lowest AIC. Built with statsmodels.

### Performance
AIC around 12044, log likelihood -6014. Residuals have no autocorrelation but aren’t normally distributed, so intervals may be off. Tested on 2009-2011 data.

### Limitations
Residuals not normal, only total sales predicted, no seasonality or product info. Not good for long-term or sudden changes.

### Trade-offs
Simple and easy, good short-term forecast but no external factors or seasonality considered.



## 3. Logistic Regression - Predicting Customer Label
### Model Description
I built a multinomial logistic regression model to predict customer return types based on features like quantity, price, invoice details, and customer purchase and return behavior. The data was prepared using one-hot encoding for categories, robust scaling for numbers, and PCA to reduce feature size.

### Input
The model uses features like quantity, price, customer ID, invoice info, day/month/year extracted from dates, and customer stats like total purchases, returns, visits, and return ratio.

### Output
It predicts customer groups: First-Time Customer, Non-Returner, Genuine Returner, High Returner, and Habitual Returner, turned into numbers for the model.

### Model Architecture
The model is a multinomial logistic regression trained with the lbfgs solver. I balanced the dataset using SMOTE to handle uneven class sizes. Preprocessing includes encoding, scaling, and PCA with 7 components.

### Performance
With SMOTE and enough training iterations, the model gives almost perfect scores on the test set. Without balancing or fewer iterations, it struggles, especially with smaller classes. Increasing iterations helps improve results. Overall accuracy is about 93%.

### Limitations
Since the model uses synthetic data from SMOTE, it might overfit or not reflect real-world cases perfectly. Some smaller classes still have low precision, meaning more false positives. The model depends on good features and might not work well if data changes.

### Trade-offs
SMOTE improves recall for small classes but may cause overfitting. More training time improves performance. The model focuses more on catching positives even if it means more false alarms, which could be okay depending on the goal.


## 4. Random Forest Tree – Predicting Customer Label
### Model Description
I built a Random Forest classifier to predict customer returner labels using features like total purchases, returns, visits, and encoded categorical variables. The input data was balanced with SMOTE and categorical features were one-hot encoded before training.

### Input
The model takes in numerical features such as total purchases, total returns, total visits, and encoded categorical features from customer data.

### Output
It predicts customer groups: First-Time Customer, Non-Returner, Genuine Returner, High Returner, and Habitual Returner, encoded as numeric labels.

### Model Architecture
The model is a Random Forest with 200 trees, no max depth limit, minimum samples split 2, and minimum samples leaf 1. Class weights were balanced to help with uneven class sizes.

### Performance
The best model got around 80% accuracy on the test set. Precision was perfect (1.00) for most classes, but recall varied — some smaller classes had low recall (like 0.29). The weighted F1-score was about 0.78, showing decent balance but room to improve recall on minority classes.

### Limitations
The model struggles with recall for small classes, meaning it misses some customers in these groups. SMOTE balancing helps but might cause some overfitting or unrealistic examples.

### Trade-offs
Using class weights and SMOTE helps with minority class recall but can hurt precision slightly and increase training time. The model focuses more on precision, so it might miss some true positives.



## 5. XGBoost
### Model Description
I used XGBoost to predict customer returner labels from the dataset, utilizing features like total purchases, returns, visits, and categorical data. The model was tuned with RandomizedSearchCV to explore a broad set of hyperparameters.

### Input
The model takes in both numerical and categorical features, including total purchases, total returns, total visits, and customer labels (after one-hot encoding categorical variables).

### Output
The model predicts customer returner labels, which are numerically encoded as: First-Time Customer, Non-Returner, Genuine Returner, High Returner, and Habitual Returner.

### Model Architecture
The model uses XGBoost with 300 trees, a maximum depth of 10, a learning rate of 0.1, and column subsample of 0.8. It uses `mlogloss` as the evaluation metric and is designed for multiclass classification.

### Performance
After tuning, the XGBoost model achieved an accuracy of 58% on the test set. The precision was high for most classes, but the recall was low for smaller classes, particularly class 0 and 1. The confusion matrix showed a dominant prediction for class 4, with other classes being heavily misclassified as class 4.

### Limitations
The model suffers from class imbalance, with class 4 being predicted overwhelmingly. The recall for the minority classes (such as class 0, 1, 3) is low, meaning the model misses a large number of true positives in these groups. SMOTE was used to balance the dataset, but the model's performance on minority classes could still be improved.

### Trade-offs
The current model prioritizes precision for class 4, but this comes at the expense of recall for smaller classes. Increasing model complexity by adding more trees or tuning further could help, but the trade-off might be an increase in training time and overfitting. Balancing the dataset further or using different techniques like class weighting could improve performance for the smaller classes.



## 6. KMeans – Customer Segmentation
### Model Description
The model applies KMeans clustering to customer data to identify customer behavior patterns based on their purchase activity. The data features include total purchases, returns, visits, and total spent per customer, which were preprocessed using PCA to reduce dimensionality to 2 components.

### Input
The input features consist of numerical data representing customer purchases, returns, total visits, and total spending. The data is first scaled using a RobustScaler to handle outliers.

### Output
The output is a set of cluster labels, where each customer is assigned to one of the 4 clusters, indicating their behavior pattern.

### Model Architecture
The model uses KMeans clustering with 4 clusters. The optimal number of clusters was determined using the elbow method with inertia and confirmed by a high Silhouette Score of 0.97, indicating well-separated clusters.

### Performance
The clustering model was evaluated using the Silhouette Score, which measures the quality of clusters. A score of 0.97 indicates that the clustering is highly well-defined, with customers in each cluster being more similar to their own group than to other groups.

### Limitations
KMeans assumes that clusters are spherical and equally sized, which might not always be true in real-world datasets. The number of clusters (k=4) was chosen based on the elbow method, but this might not always be optimal for different datasets.

### Trade-offs
Choosing the number of clusters is a trade-off. While 4 clusters resulted in a good performance based on the Silhouette Score, further tuning or trying different clustering algorithms could reveal more nuanced customer behavior patterns.



## 7. DBSCAN
### Model Description
The model applies DBSCAN clustering to customer data to identify customer behavior patterns based on their purchase activity. The data features include total purchases, returns, visits, and total spent per customer, which were preprocessed using PCA to reduce dimensionality to 4 components.

### Input
The input features consist of numerical data representing customer purchases, returns, total visits, and total spending. The data is scaled using StandardScaler.

### Output
The output is a set of cluster labels, where each customer is assigned to a cluster or marked as noise (if they do not fit well into any cluster).

### Model Architecture
The model uses DBSCAN clustering with eps = 0.4 and min_samples = 6. The optimal values for these parameters were determined based on the highest silhouette score, which was 0.51. The silhouette score indicates that the clustering is fairly well-defined, though there is still some noise.

### Performance
The clustering model was evaluated using the Silhouette Score, which measures the quality of clusters. A silhouette score of 0.51 indicates that the clustering is somewhat well-defined, with a reasonable separation between clusters, though there is still some noise.

