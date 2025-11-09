# MSCS_634_Lab_2 – KNN vs Radius Neighbors on Wine Dataset

**Name:** Ajal RC 
**Course:** MSCS 634 – Big Data and Data Mining  

## 1. Purpose

The purpose of this lab is to compare the performance of two instance-based classifiers, K-Nearest Neighbors (KNN) and Radius Neighbors (RNN), on the Wine dataset from the `sklearn` library. The lab explores how different parameter choices (k values and radius values) affect classification accuracy, and how these choices influence model behavior.

## 2. Dataset

- **Source:** `sklearn.datasets.load_wine`  
- **Samples:** 178 wine instances  
- **Features:** 13 numeric chemical properties (such as alcohol, malic acid, color intensity, and proline)  
- **Classes:** 3 wine categories

The dataset was split into 80% training data and 20% testing data using `train_test_split` with stratification to preserve class proportions.

## 3. Methods

### K-Nearest Neighbors (KNN)

- Tested k values: 1, 5, 11, 15, 21  
- For each k:
  - Trained `KNeighborsClassifier` on the training set.
  - Evaluated accuracy on the test set using `accuracy_score`.
  - Stored and plotted the accuracy against k.

### Radius Neighbors (RNN)

- Tested radius values: 350, 400, 450, 500, 550, 600  
- For each radius:
  - Trained `RadiusNeighborsClassifier` with `outlier_label='most_frequent'`.
  - Evaluated accuracy on the test set.
  - Stored and plotted the accuracy against radius.

## 4. Key Insights

- KNN achieved higher accuracy overall compared to RNN for the tested parameter ranges.  
- Small k (such as k = 1) was more sensitive to noise, while medium values of k provided a better balance between stability and flexibility.  
- RNN accuracy tended to decrease as the radius increased, likely because larger radius values included many neighbors from different classes, reducing the model’s ability to separate them.  
- This experiment shows that parameter tuning is critical: good k or radius choices can significantly improve performance.

## 5. When to Use KNN vs RNN

- **KNN** is preferable when:
  - The dataset is relatively small and clean.
  - We want a simple, intuitive model.
  - It is easy to search over a range of k values.

- **RNN** may be preferable when:
  - We care about a fixed distance threshold in feature space.
  - Data points are unevenly distributed, and a fixed neighborhood radius is more meaningful than a fixed neighbor count.

## 6. Challenges and Decisions

- Choosing appropriate radius values for RNN was less intuitive than selecting k for KNN. Too small a radius produced unstable predictions, while too large a radius mixed many points from different classes.
- No additional preprocessing (such as scaling) was applied in this lab to keep the focus on understanding KNN and RNN behavior directly on the raw Wine features.
- Visualizing accuracy trends was helpful to clearly see how each parameter affected performance.

## 7. Files in Repository

- `lab2.ipynb` – Jupyter Notebook containing the full code, visualizations, and analysis.  
- `README.md` – This documentation file.

## 8. Reference

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
