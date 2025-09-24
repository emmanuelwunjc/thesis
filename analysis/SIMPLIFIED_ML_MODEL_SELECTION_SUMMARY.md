# Simplified Machine Learning Model Selection Summary
============================================================

## Key Findings

### Best Model by R2
- **Model**: Random Forest
- **Features**: diabetic, privacy_caution_index, age_continuous, region_numeric, has_insurance, male
- **Test R²**: -0.1239
- **Test MSE**: 0.0403
- **Test MAE**: 0.1588

### Best Model by WEIGHTED_R2
- **Model**: Random Forest
- **Features**: diabetic, privacy_caution_index, age_continuous, region_numeric, has_insurance, male
- **Test R²**: -0.1239
- **Test MSE**: 0.0403
- **Test MAE**: 0.1588

### Best Model by MSE
- **Model**: Random Forest
- **Features**: diabetic, privacy_caution_index, age_continuous, region_numeric, has_insurance, male
- **Test R²**: -0.1239
- **Test MSE**: 0.0403
- **Test MAE**: 0.1588

### Best Model by MAE
- **Model**: Random Forest
- **Features**: diabetic, privacy_caution_index, age_continuous, education_numeric, male, race_numeric
- **Test R²**: -0.1439
- **Test MSE**: 0.0411
- **Test MAE**: 0.1576

## Top 10 Model Combinations

1. **Random Forest** (R² = -0.1239)
   - Features: diabetic, privacy_caution_index, age_continuous, region_numeric, has_insurance, male
   - MSE: 0.0403, MAE: 0.1588

2. **Random Forest** (R² = -0.1404)
   - Features: diabetic, privacy_caution_index, age_continuous, has_insurance, male, race_numeric
   - MSE: 0.0409, MAE: 0.1599

3. **Random Forest** (R² = -0.1412)
   - Features: diabetic, privacy_caution_index, age_continuous, urban, has_insurance, male
   - MSE: 0.0410, MAE: 0.1600

4. **Random Forest** (R² = -0.1413)
   - Features: diabetic, privacy_caution_index, age_continuous, has_insurance, received_treatment, male
   - MSE: 0.0410, MAE: 0.1599

5. **Random Forest** (R² = -0.1413)
   - Features: diabetic, privacy_caution_index, age_continuous, has_insurance, stopped_treatment, male
   - MSE: 0.0410, MAE: 0.1599

6. **Random Forest** (R² = -0.1439)
   - Features: diabetic, privacy_caution_index, age_continuous, education_numeric, male, race_numeric
   - MSE: 0.0411, MAE: 0.1576

7. **Random Forest** (R² = -0.1440)
   - Features: diabetic, privacy_caution_index, age_continuous, education_numeric, male
   - MSE: 0.0411, MAE: 0.1577

8. **Random Forest** (R² = -0.1446)
   - Features: diabetic, privacy_caution_index, age_continuous, education_numeric, urban, male
   - MSE: 0.0411, MAE: 0.1576

9. **Random Forest** (R² = -0.1446)
   - Features: diabetic, privacy_caution_index, age_continuous, education_numeric, received_treatment, male
   - MSE: 0.0411, MAE: 0.1576

10. **Random Forest** (R² = -0.1446)
   - Features: diabetic, privacy_caution_index, age_continuous, education_numeric, stopped_treatment, male
   - MSE: 0.0411, MAE: 0.1576

## Model Performance Summary

| Model | Mean R² | Std R² | Max R² | Mean MSE | Min MSE |
|-------|---------|--------|--------|----------|---------|
| ('test_r2', 'mean') | N/A | N/A | N/A | N/A | N/A |
| ('test_r2', 'std') | N/A | N/A | N/A | N/A | N/A |
| ('test_r2', 'max') | N/A | N/A | N/A | N/A | N/A |
| ('test_mse', 'mean') | N/A | N/A | N/A | N/A | N/A |
| ('test_mse', 'std') | N/A | N/A | N/A | N/A | N/A |
| ('test_mse', 'min') | N/A | N/A | N/A | N/A | N/A |
| ('test_mae', 'mean') | N/A | N/A | N/A | N/A | N/A |
| ('test_mae', 'std') | N/A | N/A | N/A | N/A | N/A |
| ('test_mae', 'min') | N/A | N/A | N/A | N/A | N/A |

## Key Insights

1. **Automatic Model Selection**: ML methods automatically find optimal feature combinations
2. **Diabetes and Privacy Always Included**: Ensures core variables are in every model
3. **Comprehensive Testing**: Multiple algorithms tested with various feature combinations
4. **Performance Optimization**: Best models identified by multiple metrics
5. **Feature Importance**: Understanding which features contribute most to performance

## Methodology

- **Algorithms**: Random Forest, Linear Regression, Ridge, Lasso
- **Feature Selection**: All combinations of 3-6 features (diabetes + privacy always included)
- **Evaluation**: Train/test split with cross-validation
- **Metrics**: R², MSE, MAE for comprehensive evaluation
- **Total Combinations**: 255
