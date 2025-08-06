# Data Card for Online Retail-II Data Set

## Source:
- **Original source**: UCI Machine Learning Repository - Online Retail II
- **Where I downloaded it from**: Kaggle – Online Retail II Dataset [(Link)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
- **License**: Open dataset, free for academic and research use

## Motivation:
### Purpose:
The dataset was created to help researchers, students, and analysts study customer transaction patterns. It supports projects like customer segmentation, return behavior classification, sales forecasting, and anomaly detection.

### Creator:
The dataset was made available by the UCI Machine Learning Repository.

### Founded by:
The original source is not clearly mentioned, but the dataset is hosted and maintained for academic and research purposes by the University of California, Irvine.

## Composition:
### What the data represents:
Each row is a transaction record, showing purchases or returns made by a customer at a particular time.

### Number of records:
Around 1.06 million rows, comes from two years of data (2009-2010 and 2010-2011).

### Types of Instances:
- **Invoice**: Numerical
- **StockCode**: Object (combination of numerical and character)
- **Description**: Object (product description)
- **Price**: Numerical (positive and negative price)
- **Quantity**: Numerical (both positive and negative quantity)
- **InvoiceDate**: Numerical (yyyy-mm-dd hh:mm:ss)
- **Customer ID**: Numerical
- **Country**: Object

### Missing Data:
- **Customer ID**: 243,007 entries have missing/null customer ID.
- **Description**: 4,382 entries have missing/null description.

### Target:
No target variable.

### Confidential Data:
No personal information like names or contact details included.

## Collection Process:
### How data was acquired:
The data was likely from a UK-based non-store online retailer.

### Sampling:
No sampling - Complete dataset for the given time range.

### Time frame:
From December 2009 to December 2011.

### Pre-processing/Cleaning/Labelling:
#### Was cleaning done?
Yes, it is a messy dataset, so I cleaned and engineered the data to get it ready for modeling.

#### Steps done:
- Removed rows with null/missing values, duplicated entries.
- Removed rows with negative price, negative quantity without proper invoice ('C').
- Removed rows with negative quantities that are not matched with positive quantities (looks like returns made without purchase).
- Data type of InvoiceDate changed from numerical to Date and Time data type.
- Outliers (quantities <10% and >90%) were removed.
- Created `TotalPrice = Quantity * Price`.
- Aggregated customer-level data:
    - TotalPurchases, TotalReturns, TotalVisits.
- Added customer return behavior labels:
    - First-Time, Non-Returner, Genuine Returner, High-Returner, Habitual Returner.
- Scaled the numerical features, one-hot encoded the categorical features, and label encoded the target.

#### Raw data saved?
Yes, I kept a copy of the original data separate from the processed version.

## Potential Risks, Biases, and Mitigation:
1. **Transaction-based labeling**: The dataset only includes transaction-level data like quantity, returns, and visits. There is no personal information. Any labeling of customers (e.g., habitual returner or non-returner) is purely based on transaction patterns, not who they are as individuals.
   
2. **Unexplained returns**: Return behavior might be influenced by many factors, such as faulty products, delivery issues, or dissatisfaction, but the dataset does not tell us why items were returned. This could lead to biases if labeling is used incorrectly.

3. **Dropped data**: Around 25% of entries have been dropped, and I used approximately 3.5% of the entries due to RAM issues. Thus, we do not always get the full picture of the customer journey. Models relying heavily on aggregated customer behavior could be skewed or incomplete.

4. **Returner labels**: The returner labels (e.g., First-Time, Genuine Returner, Habitual Returner) are for learning and analysis only, not for real-world decision-making unless fully validated.

5. **Fairness and imbalance**: Models using this data should be tested for imbalance and fairness, especially since some groups (e.g., Habitual Returners) are very small in number.

6. **Avoid customer scoring or punishment systems**: Be careful not to use this dataset to build customer scoring or punishment systems. There isn’t enough context behind why behaviors occurred.

7. **Geographic limitations**: The data comes from a UK-based non-store online retailer. Any model built on this might not perform the same way for retailers, regions, or industries outside of the UK.
