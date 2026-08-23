# Bengaluru House Price Prediction

A machine learning project predicting house prices in Bengaluru using classic regression models, built and trained on Google Colab.

## Problem Statement

Given property details (location, square footage, number of bedrooms/bathrooms/balconies), predict the sale price of a house in Bengaluru. This is a supervised regression problem trained on real, messy, real-world listing data.

## Dataset

- **Source:** [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) (Kaggle, by amitabhajoy)
- **Raw size:** 13,320 rows, 9 columns
- **Final cleaned size:** 9,608 rows, 235 features (after cleaning, deduplication, and outlier removal)

## Approach

### 1. Data Cleaning
- Dropped low-value / heavily-missing columns: `area_type`, `society` (41% missing), `availability`
- Imputed `balcony` nulls with median rather than dropping rows (weak feature, moderate missing %, didn't want to lose otherwise-valid rows)
- Dropped remaining small-count nulls in `location`, `size`, `bath`
- Extracted numeric `bhk` from text (`"2 BHK"` → `2`)
- Parsed `total_sqft`, which contained three different formats: plain numbers, ranges (`"1200-1400"`), and mixed units (`Sq. Meter`, `Perch`) — converted ranges and Sq. Meter properly, dropped the small number of unrecoverable/rare-unit rows (29 rows, ~0.2%)
- Removed 751 duplicate rows

### 2. Feature Engineering
- Created `price_per_sqft` (used only for outlier detection, dropped before modeling to avoid data leakage)
- Collapsed 1,287 raw location values down to 232 by grouping locations with ≤10 listings into an `"other"` bucket, to keep one-hot encoding manageable

### 3. Outlier Removal
- Removed listings with implausible sqft-per-bedroom ratios (< 300 sqft/bhk)
- Removed price-per-sqft outliers *within each location group* (more than 1 std dev from that location's mean) — done per-location because a price that's normal in a premium area is an outlier in a budget one
- Removed listings with unrealistic bathroom counts (bath ≥ bhk + 2)

### 4. Encoding
- One-hot encoded `location` (232 categories), dropping the `"other"` column to avoid the dummy variable trap

### 5. Modeling
Trained and compared 6 models: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost.

Tree-based models initially overfit badly on default settings (e.g. Decision Tree: 0.984 train R² vs 0.605 test R²). Applied hyperparameter tuning (manual constraints, then `RandomizedSearchCV`) to reduce overfitting.

### Final Results (Test Set)

| Model | Test R² | RMSE |
|---|---|---|
| **Linear Regression** | **0.757** | **44.73** |
| Ridge | 0.753 | 45.08 |
| Decision Tree (tuned) | 0.740 | 46.28 |
| Random Forest (tuned) | 0.685 | 50.89 |
| XGBoost (tuned) | 0.632 | 55.02 |
| Lasso | 0.620 | 55.89 |

**Linear Regression was the best-performing model**, even after tuning the tree-based models with `RandomizedSearchCV`.

## Key Finding & Limitation

Tree-based models (Random Forest, XGBoost) underperformed Linear Regression on this dataset. The likely cause: 231 of the 235 features are sparse one-hot encoded location columns. Tree-based splits struggle to efficiently use many sparse binary columns, while linear models handle this pattern naturally — each location simply gets its own additive coefficient. This is a known limitation of one-hot encoding with high-cardinality categorical features, not a modeling mistake.

**A possible future improvement:** replace one-hot encoding with target/mean encoding for `location`, or use a model built to handle categorical features natively (e.g. CatBoost), which may allow tree-based models to outperform the linear baseline.

The model also **underpredicts luxury properties** (actual price > ~₹1,000 lakhs) — likely due to few luxury samples in the training data, and the absence of features like builder reputation, amenities, or view, which matter more at the high end of the market.

## Usage

```python
predict_price('Whitefield', total_sqft=1200, bath=2, balcony=1, bhk=2)
# Predicted price: ₹64.09 lakhs
```

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, GridSearchCV/RandomizedSearchCV)
- XGBoost
- Matplotlib
- Google Colab

## What I'd Do Differently Next Time

- Try target/mean encoding for `location` instead of one-hot, to give tree-based models a fairer chance
- Add more features if available (amenities, age of property, floor number) to help with luxury property prediction
- Use `RandomizedSearchCV` from the start instead of full `GridSearchCV`, for faster iteration
