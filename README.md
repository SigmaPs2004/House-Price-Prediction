# Bengaluru Housing Market & Price Prediction

An analysis of Bengaluru's residential real estate market, combining SQL-based market benchmarking with a Python ML pipeline to identify what drives property prices and estimate fair pricing from listing attributes. Built and trained on Google Colab and Microsoft SQL Server Management Studio.

## Business Question

How should a property in Bengaluru be priced, and what actually drives that price — location, size, or bedroom count?

## Dataset

- **Source:** [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) (Kaggle, by amitabhajoy)
- **Raw size:** 13,320 rows, 9 columns
- **Final cleaned size:** 9,608 rows (after cleaning, deduplication, and outlier removal)

## Approach

### 1. SQL Market Analysis (Microsoft SQL Server)
Queried the cleaned 9K+ dataset to benchmark the market before modeling:
- **Top price-per-sqft localities:** Cunningham Road, Rajaji Nagar, and Benson Town led at ₹13,000–20,000/sqft — up to 2x the citywide average
- Price distribution across BHK segments

### 2. Data Cleaning (Python)
- Dropped low-value / heavily-missing columns: `area_type`, `society` (41% missing), `availability`
- Imputed `balcony` nulls with median rather than dropping rows (weak feature, moderate missing %, didn't want to lose otherwise-valid rows)
- Dropped remaining small-count nulls in `location`, `size`, `bath`
- Extracted numeric `bhk` from text (`"2 BHK"` → `2`)
- Parsed `total_sqft`, which contained three different formats: plain numbers, ranges (`"1200-1400"`), and mixed units (`Sq. Meter`, `Perch`) — converted ranges and Sq. Meter properly, dropped the small number of unrecoverable/rare-unit rows (29 rows, ~0.2%)
- Removed 751 duplicate rows

### 3. Feature Engineering
- Created `price_per_sqft` (used only for outlier detection, dropped before modeling — this column is derived directly from `price` and would otherwise leak the target into the features)
- Collapsed 1,287 raw location values down to 232 by grouping locations with ≤10 listings into an `"other"` bucket, to keep one-hot encoding manageable

### 4. Outlier Removal
- Removed listings with implausible sqft-per-bedroom ratios (< 300 sqft/bhk)
- Removed price-per-sqft outliers *within each location group* (more than 1 std dev from that location's mean) — done per-location because a price that's normal in a premium area is an outlier in a budget one
- Removed listings with unrealistic bathroom counts (bath ≥ bhk + 2)
- Kept genuine extreme-value properties (e.g. a 12,000 sqft, 7 BHK, ₹22 crore listing) rather than dropping them — they're real, valid data, not errors, and dropping them would only flatter the evaluation

### 5. Multicollinearity Check
Ran VIF (Variance Inflation Factor) on the core numeric features and found `bath` and `bhk` severely collinear (VIF 35+ each, correlation 0.87 — bathrooms scale almost linearly with bedrooms in this market). Dropped `bath`, which:
- Reduced VIF to acceptable levels (all under 10)
- Corrected `bhk`'s regression coefficient from a nonsensical **negative** value to a logically correct **positive** one
- Left predictive performance essentially unchanged (R² 0.757 → 0.757) — confirming `bath` carried no unique signal beyond what `bhk` already captured

**Note:** fixing multicollinearity improved coefficient *reliability*, not predictive *accuracy* — these are different goals, and it's worth being precise about which one a given fix addresses.

### 6. Encoding
One-hot encoded `location` (232 categories), dropping the `"other"` column to avoid the dummy variable trap.

### 7. Modeling & Evaluation
Trained and compared 6 regression models: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost.

Tree-based models initially overfit badly on default settings (e.g. Decision Tree: 0.984 train R² vs 0.605 test R²). Applied hyperparameter tuning (`RandomizedSearchCV`) to reduce overfitting.

**Final Results (single 80/20 test split):**

| Model | Test R² | RMSE |
|---|---|---|
| **Linear Regression** | **0.757** | **44.74** |
| Ridge | 0.753 | 45.08 |
| Random Forest | 0.677 | 51.56 |
| XGBoost | 0.661 | 52.84 |
| Lasso | 0.623 | 55.68 |
| Decision Tree | 0.618 | 56.05 |

**5-Fold Cross-Validation (Linear Regression, more robust estimate):**
`[0.535, 0.673, 0.597, 0.780, 0.726]` → **Mean R² = 0.662, Std = 0.088**

The single test-split score (0.757) is somewhat more favorable than the honest cross-validated mean (0.662) — the CV result is reported here because a single split can get lucky, and the ±0.088 spread across folds is itself informative: performance is sensitive to which rare locations land in train vs. test, a natural consequence of the long-tail location distribution.

## Key Findings

**1. Location is the dominant price driver — not size or bedroom count.** Regression coefficient analysis (post multicollinearity fix) found:
- Location: **₹531 lakh** spread between the highest- and lowest-priced localities
- Property size (`total_sqft`): **₹60 lakh** spread per standard deviation
- Bedroom count (`bhk`): **₹1.76 lakh** spread per standard deviation

Location's effect on price is **~9x larger** than property size, and dramatically larger than bedroom count.

**2. The model reliably predicts mid-market properties, but underpredicts luxury listings** (actual price > ~₹1,000 lakh) — likely due to few luxury samples in training data and the absence of features like builder reputation, amenities, or exact street prestige, which matter more at the high end.

## Data & Model Validation Performed
- Checked for and removed data leakage (accidentally leaving the derived `price_per_sqft` column in the feature set)
- VIF and correlation-based multicollinearity diagnosis and fix
- Train/test performance gap checks on every tuned model, to catch overfitting rather than trusting test R² alone
- Cross-validation to check single-split result stability
- Residual analysis to check for systematic prediction errors
- Edge-case input testing on the prediction function (confirmed the model extrapolates poorly on unrealistic sqft/bhk combinations outside its training distribution — a documented limitation, not a bug)

## Usage

```python
predict_price_fixed('Whitefield', total_sqft=1200, balcony=1, bhk=2)
# Predicted price: ₹64.43 lakhs
```

## Tech Stack

- SQL (Microsoft SQL Server / T-SQL)
- Python, pandas, NumPy
- scikit-learn (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, RandomizedSearchCV)
- XGBoost
- statsmodels (VIF, Cook's Distance)
- Matplotlib
- Google Colab
