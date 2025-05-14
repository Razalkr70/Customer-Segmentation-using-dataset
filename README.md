# 🛍️ Customer Segmentation using K-Means Clustering

This project performs customer segmentation on a mall customer dataset using the K-Means clustering algorithm. It identifies groups based on features like age, income, and spending score, and visualizes the clusters using pair plots and PCA in 3D.

## 📌 Overview

Customer segmentation is a key technique in marketing and business analytics. In this project, the K-Means algorithm is applied to group customers based on their demographics and spending patterns.

### ✨ Features

- Data preprocessing and feature scaling  
- Gender encoding  
- Elbow method to determine optimal `k`  
- Cluster formation using K-Means  
- Cluster-wise statistical summary  
- Visualizations using seaborn and matplotlib  
- 3D PCA for better insight into clusters  
- Customer labeling using custom logic  

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- PCA (Principal Component Analysis)  
 

## 📊 How It Works

1. Dataset is preprocessed and gender is encoded.
2. Elbow method is used to determine the optimal number of clusters.
3. K-Means is applied to group customers.
4. Cluster visualization using seaborn and PCA.
5. Each cluster is labeled with intuitive names like "Young Spenders", "Savers", etc.

## 📂 Dataset

`Mall_Customers.csv` should be in your working directory. It contains:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## 🚀 Run the Code

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python customer_segmentation.py
```
### 📈 Sample Output
- Cluster visualization via pairplots
- 3D PCA cluster plot
- Cluster statistics


