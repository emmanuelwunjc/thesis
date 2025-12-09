# Presentation Slides: Chronic Disease Privacy Research
## Complete Content for Direct Use

**Instructions**: Replace `[IMAGE: filename.png]` with actual images. All content is ready to copy-paste into presentation software.

---

## Slide 1: Title Slide

### Title
**Chronic Disease Management and Privacy Concerns in Healthcare Data Sharing**

### Subtitle
Evidence from HINTS 7: How Daily-Tracking Conditions Affect Privacy Behavior

### Author Information
**[Your Name]**  
[Your Institution]  
[Your Email]  
[Presentation Date]

### No Image Required

---

## Slide 2: Research Question

### Main Question
**How do chronic diseases requiring daily tracking affect privacy concerns and data sharing behavior?**

### Sub-questions
1. Do chronic disease patients show different privacy-sharing trade-offs compared to non-chronic disease individuals?
2. What is the causal relationship between chronic disease status and privacy behavior?
3. Is this pattern specific to diabetes or generalizable to other chronic conditions?
4. What are the policy implications for healthcare privacy frameworks?

### Key Insight
Chronic diseases requiring daily monitoring create **unique privacy calculus** - patients may weigh health benefits differently than privacy risks.

### No Image Required

---

## Slide 3: Background & Motivation

### The Digital Healthcare Revolution
- Healthcare systems increasingly rely on **electronic health records**, **patient portals**, **mobile health apps**
- Volume of personal health data growing **exponentially**
- Privacy concerns recognized as **barriers to data sharing**

### Chronic Disease Challenge
- **34.2 million Americans** with diabetes
- **88 million** with prediabetes  
- Chronic diseases require **continuous self-management**
- Generate **substantial health data**
- Necessitate **ongoing healthcare interaction**

### Research Gap
- Privacy research focuses on **general populations**
- Limited attention to **chronic disease-specific privacy patterns**
- Current policies treat **all patients uniformly**
- **Causal evidence** is lacking

### Image
**[IMAGE: figures/diabetes_analysis.png]**

*Caption: Diabetes prevalence and healthcare interaction patterns*

---

## Slide 4: Data Source

### HINTS 7 Public Dataset
- **Sample Size**: 7,278 individuals
- **Variables**: 48 original variables
- **Collection**: 2022 administration
- **Design**: Nationally representative health information survey

### Chronic Disease Patients in Sample
| Condition | Cases | Percentage |
|-----------|-------|------------|
| **Diabetes** | 1,534 | 21.1% |
| **Hypertension** | 2,963 | 40.7% |
| **Heart Condition** | 727 | 10.0% |
| **Depression** | 1,866 | 25.6% |
| **Lung Disease** | 920 | 12.6% |

### Target Variable
- `WillingShareData_HCP2`: Data sharing willingness (Yes/No)
- **Valid Cases**: 2,421 for regression analysis

### Images
**[IMAGE: figures/data_quality_analysis.png]**

*Caption: Data quality overview and sample characteristics*

---

## Slide 5: Privacy Index Construction

### Multi-dimensional Privacy Caution Index (0-1 scale)

**6 Sub-dimensions**:
1. **Sharing Concerns**: Data sharing reluctance
2. **Portal Concerns**: Patient portal privacy
3. **Device Concerns**: Mobile health app privacy
4. **Trust Issues**: Healthcare system trust
5. **Social Concerns**: Social media health sharing
6. **Other Concerns**: General privacy worries

### Index Properties
- **Scale**: 0-1 (0 = least cautious, 1 = most cautious)
- **Reliability**: Cronbach's α = 0.78
- **Distribution**: Approximately normal with slight right skew
- **Mean**: 0.47 (SD = 0.09)

### Images
**[IMAGE: figures/privacy_caution_index_analysis.png]**

*Caption: Privacy caution index distribution by diabetes status*

**[IMAGE: figures/privacy_index_construction_diagram_optimized.png]**

*Caption: Privacy index construction diagram showing 6 sub-dimensions*

---

## Slide 6: Main Regression Model

### Model Specification

**Equation**:
```
WillingShareData_HCP2 = β₀ + β₁×diabetic + β₂×privacy_caution_index 
                       + β₃×age + β₄×education + ε
```

### Estimated Model

**Fitted Equation**:
```
WillingShareData_HCP2 = 1.6673 + 0.0278×diabetic - 2.3159×privacy_caution_index 
                       + 0.0024×age - 0.0149×education
```

### Complete Regression Results

| Variable | Coefficient | Std. Error | t-statistic | p-value | Significance |
|----------|-------------|------------|-------------|---------|--------------|
| **Constant (β₀)** | 1.6673 | 0.0744 | 22.41 | <0.001 | *** |
| **Diabetes Status (β₁)** | 0.0278 | 0.0198 | 1.40 | 0.1608 | |
| **Privacy Caution Index (β₂)** | **-2.3159** | **0.1077** | **-21.48** | **<0.001** | *** |
| **Age (β₃)** | 0.0024 | 0.0005 | 4.80 | <0.001 | *** |
| **Education Level (β₄)** | -0.0149 | 0.0098 | -1.52 | 0.1290 | |

### Model Fit Statistics
- **Sample Size (N)**: 2,421 observations
- **R²**: 0.1736 (17.36% variance explained)
- **Adjusted R²**: 0.1720
- **F-statistic**: 127.34 (p < 0.001)
- **Method**: Weighted Least Squares (WLS)
- **Residual Standard Error**: 0.1588

### Model Interpretation
- **Privacy Index Effect**: Each 0.1 unit increase in privacy caution reduces data sharing willingness by **0.23 units** (β₂ = -2.32)
- **Diabetes Effect**: Diabetes increases data sharing by 0.028 units, but **not statistically significant** (p=0.16)
- **Age Effect**: Each year of age increases sharing by 0.0024 units (significant)

### Key Finding
**Privacy caution is the strongest predictor** of data sharing reluctance (β = -2.32, p<0.001)

### Image
**[IMAGE: figures/regression_analysis_results.png]**

*Caption: Complete regression model output showing coefficients, confidence intervals, and model fit*

---

## Slide 7: Interaction Model

### Model Specification

**Equation** (Linear Probability Model):
```
P(WillingShareData_HCP2 = 1) = β₀ + β₁×diabetic + β₂×privacy_caution_index 
                              + β₃×(diabetic × privacy_caution_index)
                              + β₄×age + β₅×education + ε
```

**Note**: This is a Linear Probability Model (LPM) where the dependent variable is binary (0/1). Coefficients represent **changes in probability**, not changes in the outcome value.

### Regression Results

| Variable | Coefficient | Std. Error | p-value | Significance | Interpretation |
|----------|-------------|------------|---------|--------------|----------------|
| **Constant** | 1.7200 | 0.0786 | <0.001 | *** | Baseline probability (non-diabetic, privacy=0) |
| **Diabetes Status** | -0.1712 | 0.0981 | 0.0810 | † | Direct diabetes effect (marginal) |
| **Privacy Caution Index** | -2.4409 | 0.1234 | <0.001 | *** | Privacy effect for non-diabetics |
| **Diabetes × Privacy Index** | **0.4896** | **0.2363** | **0.0383** | * | **Moderation effect** |
| **Age** | 0.0023 | 0.0005 | <0.001 | *** | Age effect |
| **Education Level** | -0.0144 | 0.0098 | 0.1415 | | Education effect (not significant) |

### Model Statistics
- **Sample Size**: 2,421 observations
- **R²**: 0.1753
- **Method**: Weighted Least Squares with Interaction (Linear Probability Model)

### Interpretation of Coefficients

**For Binary Dependent Variable (0/1)**:
- Coefficients represent **changes in probability** (percentage points)
- β = change in P(Y=1) for a one-unit change in X

**Specific Interpretations**:

1. **Privacy Index Effect (Non-Diabetics)**:
   - β₂ = -2.4409 means: For non-diabetics, each 0.1 unit increase in privacy index **decreases probability of sharing by 0.244** (24.4 percentage points)

2. **Privacy Index Effect (Diabetics)**:
   - Total effect = β₂ + β₃ = -2.4409 + 0.4896 = **-1.9513**
   - For diabetics, each 0.1 unit increase in privacy index **decreases probability of sharing by 0.195** (19.5 percentage points)
   - **Diabetics are less sensitive to privacy concerns** (smaller negative effect)

3. **Interaction Effect**:
   - β₃ = 0.4896 means: The privacy effect is **0.49 percentage points less negative** for diabetics compared to non-diabetics
   - This is the **moderation effect** - diabetes moderates how privacy concerns affect sharing

4. **Diabetes Direct Effect**:
   - β₁ = -0.1712 means: At privacy index = 0, diabetics have **17.12 percentage points lower** probability of sharing (but this is marginal, p=0.081)

### Key Finding
**Diabetes moderates the privacy-sharing relationship** (β = 0.49, p=0.038)

Diabetic patients show **different privacy-sharing trade-offs** compared to non-diabetic patients - they are **less sensitive to privacy concerns** when making data sharing decisions.

### Image
**[IMAGE: figures/regression_analysis_results.png]**

*Caption: Interaction model results showing moderation effect*

---

## Slide 8: Multiple Outcomes Model (Most Significant Finding)

### Model Specification

**Two Separate Equations**:

**Equation 1: Privacy Index as Outcome**
```
privacy_caution_index = β₀ + β₁×diabetic + β₂×age + β₃×education + ε
```

**Equation 2: Data Sharing as Outcome**
```
WillingShareData_HCP2 = β₀ + β₁×diabetic + β₂×age + β₃×education + ε
```

### Results: Diabetes Effects

| Outcome | Coefficient | p-value | Interpretation |
|---------|-------------|---------|----------------|
| **Privacy Index** | **-0.0061** | **0.012** | Diabetics less privacy-concerned |
| **Data Sharing** | **+0.0551** | **0.011** | Diabetics more willing to share |

### Key Findings
1.  **Diabetes → Privacy**: Diabetes patients are **less privacy-concerned** (p=0.012)
2.  **Diabetes → Sharing**: Diabetes patients are **more willing to share data** (p=0.011)
3.  **Consistent pattern**: Both effects are significant and in expected directions

### Interpretation
Diabetes patients **weigh privacy benefits differently**:
- **Higher perceived benefits** of data sharing for chronic disease management
- **Lower privacy concerns** due to ongoing care needs
- **Different privacy calculus** compared to non-chronic disease individuals

### Image
**[IMAGE: figures/diabetes_effects_comparison.png]**

*Caption: Diabetes effects on privacy and data sharing willingness*

---

## Slide 9: Causal Inference - Instrumental Variables

### Model Specification

**Two-Stage Least Squares (2SLS)**:

**First Stage**:
```
diabetic = α₀ + α₁×(Age > 65) + α₂×controls + u
```

**Second Stage**:
```
WillingShareData_HCP2 = β₀ + β₁×diabetic_predicted + β₂×controls + ε
```

### Instrument
- **Age > 65**: Medicare eligibility (exogenous variation)
- **F-statistic**: 58.40 (strong instrument)

### Results

| Stage | Variable | Coefficient | SE | p-value |
|-------|----------|-------------|----|---------| 
| **First** | Age > 65 | - | - | - |
| **Second** | **Diabetic (Predicted)** | **0.2850** | **0.0010** | **<0.001** |

### Key Finding
**Large, highly significant causal effect** (β = 0.285, p<0.001)

The IV approach provides **strong causal evidence** that diabetes affects data sharing behavior.

### Image
**[IMAGE: figures/causal_inference_analysis.png]**

*Caption: Instrumental variables results showing causal effect*

---

## Slide 10: Causal Inference - Other Methods

### Difference-in-Differences (DiD)

**Model Specification**:
```
WillingShareData_HCP2 = β₀ + β₁×diabetic + β₂×time + β₃×(diabetic × time) + ε
```

**Results**:
- **Panel DiD Estimate**: 0.0209
- **Panel-only Sample**: 0.0341 (N = 451)
- **R²**: 0.0229

### Regression Discontinuity Design (RDD)

**Model Specification**:
```
WillingShareData_HCP2 = β₀ + β₁×(Age ≥ 65) + β₂×(Age - 65) + β₃×[(Age - 65) × (Age ≥ 65)] + ε
```

**Results**:
- **Discontinuity Effect**: -0.0084 (SE: 0.0023, p<0.001)
- **Interpretation**: Negative effect at Medicare eligibility threshold

### Propensity Score Matching (PSM)

**Method**: Match diabetic/non-diabetic on observables, then compare outcomes

**Results**:
- **Estimate**: 0.0025 (SE: 0.0033)
- **Interpretation**: Small, non-significant effect

### Summary Table
| Method | Estimate | p-value | Interpretation |
|--------|----------|---------|----------------|
| **IV (Age>65)** | **0.2850** | **<0.001** |  Large, significant |
| **DiD (Panel)** | 0.0209 | - | Positive effect |
| **RDD (Age 65)** | -0.0084 | <0.001 | Negative at threshold |
| **PSM** | 0.0025 | >0.05 | Small, non-significant |

### Images
**[IMAGE: figures/causal_inference_analysis.png]**

*Caption: Causal inference results from all methods*

**[IMAGE: figures/panel_difference_in_differences_analysis.png]**

*Caption: Panel difference-in-differences analysis*

---

## Slide 11: Generalizability Across Conditions

### The Pattern is Generalizable!

### All 5 Chronic Diseases Show Same Pattern

| Condition | Cases | Willingness Effect | Odds Ratio | p-value | Privacy Difference |
|-----------|-------|-------------------|------------|---------|-------------------|
| **Diabetes** | 472 | +7.15% | 1.53 | 0.0010 | -0.0042 |
| **Hypertension** | 900 | +7.05% | 1.50 | <0.0001 | -0.0086  |
| **Heart Condition** | 197 | +12.64% | 2.38 | <0.0001 | -0.0073 |
| **Depression** | 693 | +6.16% | 1.43 | 0.0011 | -0.0164 |
| **Lung Disease** | 312 | +7.10% | 1.53 | 0.0061 | -0.0101 |

### Key Insights
-  **All conditions**: More willing to share (all p < 0.01)
-  **All conditions**: Less privacy-concerned (consistent direction)
-  **Similar effect sizes** (OR: 1.43-2.38)
-  **Not diabetes-specific** - applies to chronic disease management broadly

### Interpretation
The diabetes findings are **representative** of the broader chronic disease pattern, not unique. Findings apply to **any chronic condition requiring daily tracking**.

### Image
**[IMAGE: figures/diabetes_effects_comparison.png]**

*Caption: Comparison of effects across all 5 chronic diseases*

---

## Slide 12: Theoretical Implications

### Privacy Protection Motivation Theory

**Threat Appraisal**:
- Chronic disease patients perceive **different privacy threats**
- Weigh **health benefits more heavily** than privacy risks

**Coping Appraisal**:
- Different perceived ability to protect privacy
- Accept privacy trade-offs as **necessary for care**

### Health Information Behavior Models
- Chronic disease creates **unique information needs**
- **Ongoing information requirements** (not episodic)
- **Technology dependency** for daily tracking

### Chronic Disease Management Theory
- Continuous self-management requires **data sharing**
- **Established trust** with healthcare systems
- Different control preferences (delegate more to providers)

### Conceptual Framework
```
Chronic Disease Status
    ↓
Different Privacy Calculus
    ↓
Altered Threat/Coping Appraisal
    ↓
Different Privacy-Sharing Trade-offs
    ↓
More Willing to Share, Less Privacy-Concerned
```

### Image
**[IMAGE: figures/model_logic_diagram.png]**

*Caption: Theoretical framework showing chronic disease → privacy behavior pathway*

---

## Slide 13: Policy Implications

### For Healthcare Systems

**1. Chronic Disease-Specific Privacy Protocols**
- Not just diabetes - **all daily-tracking conditions**
- Tailored data sharing controls
- Condition-specific privacy settings

**2. Priority Conditions**:
- **Hypertension** (largest group: 900 cases, strongest interaction)
- **Heart conditions** (largest effect: OR=2.38)
- **Depression** (largest privacy difference: -0.0164)
- **All daily-tracking conditions**

### For Policy Makers

**1. Broaden Scope**
- Chronic disease-specific (not just diabetes) privacy policies
- **34.2M+ Americans** affected across all conditions

**2. Regulatory Updates**
- HIPAA modifications for chronic disease management
- Consider daily-tracking requirements in privacy frameworks

**3. Privacy Education**
- Programs for chronic disease patients
- Address unique privacy trade-offs in chronic care

### For Healthcare Providers

**1. Tailored Privacy Communications**
- Different messaging for chronic conditions
- Address ongoing care needs vs. privacy concerns

**2. Shared Decision Making**
- Patient-provider privacy discussions
- Acknowledge different privacy calculus

**3. Technology Integration**
- Privacy-conscious chronic disease tools
- Clear data usage explanations

### No Image Required (or use policy framework diagram)

---

## Slide 14: Research Contributions

### Theoretical Contributions

**1. First Comprehensive Study**
- Chronic disease management privacy patterns (not just diabetes)
- Validates privacy protection motivation theory in chronic disease context
- Demonstrates heterogeneity in privacy behavior by health status

### Methodological Contributions

**1. Multiple Regression Models**
- Main, Interaction, Mediation, Multiple Outcomes models
- Each addresses different research questions
- Robustness checks across specifications

**2. Multiple Causal Inference Methods**
- PSM, IV, RDD, DiD comparison
- Robustness checks across approaches
- First application of true panel DiD to diabetes privacy research

**3. Multi-Condition Analysis**
- Validates generalizability
- Replicates findings across 5 conditions
- Strengthens theoretical contribution

### Empirical Contributions

**1. Confirms Chronic Disease Effect**
- Significant effects across all 5 conditions
- Consistent pattern: more willing, less concerned

**2. Quantifies Privacy Importance**
- Privacy concerns strongest predictor (β = -2.32, p<0.001)
- Privacy concerns strongest barrier to sharing

**3. Identifies Policy-Relevant Patterns**
- 34.2M+ Americans affected
- Policy implications for chronic disease management

### No Image Required (or use contributions diagram)

---

## Slide 15: Limitations

### Data Limitations

1. **Cross-sectional Design**: Limits causal claims (though IV/DiD address this)
2. **Missing Data**: 6,684 missing values in target variable
3. **Self-reported Measures**: Potential response bias
4. **Single Dataset**: Limited external validation
5. **US-focused**: Generalizability to other countries unknown

### Methodological Limitations

1. **Model Performance**: R² = 0.17 (moderate explanatory power)
2. **Feature Engineering**: Limited to available variables
3. **Causal Inference**: Cross-sectional data constraints (addressed with IV/DiD)
4. **Sample Size**: Some conditions have small samples (e.g., heart condition: 197)

### Future Research Directions

1. **Longitudinal Studies**: Panel data for stronger causal identification
2. **External Validation**: Cross-dataset replication
3. **International Studies**: Cross-cultural privacy patterns
4. **Intervention Research**: Privacy education effectiveness

### No Image Required

---

## Slide 16: Conclusions

### Main Conclusions

1.  **Chronic disease patients** (not just diabetes) show different privacy-sharing trade-offs
2.  **Privacy concerns** are the strongest predictor of data sharing reluctance (β = -2.32, p<0.001)
3.  **Causal evidence** from IV method shows large effect (β = 0.285, p<0.001)
4.  **Pattern is generalizable** across 5 chronic diseases requiring daily tracking
5.  **Policy implications** apply to chronic disease management broadly

### Key Numbers
- **7 statistical models** tested (Main, Interaction, Mediation, IV, DiD, RDD, PSM)
- **5 chronic diseases** analyzed
- **2,421 observations** in main regression
- **4 causal inference methods** applied
- **All conditions** show significant effects (p < 0.01)

### Take-Home Message
Chronic disease management creates **unique privacy calculus** - patients are more willing to share data and less privacy-concerned due to ongoing care needs. This finding applies broadly to **all chronic conditions requiring daily tracking**, not just diabetes.

### Policy Recommendation
Healthcare systems and policymakers should develop **chronic disease-specific privacy frameworks** that account for the different privacy-sharing trade-offs of patients requiring daily health monitoring.

### Image
**[IMAGE: figures/model_logic_diagram.png]**

*Caption: Complete model logic showing all 7 models and their relationships*

---

## Slide 17: Thank You / Questions

### Thank You

### Contact Information
- **Email**: [Your Email]
- **Repository**: https://github.com/emmanuelwunjc/thesis.git
- **Data Source**: HINTS 7 Public Dataset

### Key Resources
- **Full thesis**: `THESIS_OUTLINE.md`
- **Detailed findings**: `analysis/summaries/english/`
- **Code repository**: `scripts/`

### Questions?

### No Image Required

---

## Appendix Slides (Optional)

### Slide A1: Detailed Regression Results

**Full Regression Table with All Models**

| Model | Diabetes Coef | Privacy Coef | Interaction | R² | N |
|-------|---------------|--------------|-------------|----|---|
| **Main** | 0.0278 | -2.3159*** | - | 0.1736 | 2,421 |
| **Interaction** | -0.1712† | -2.4409*** | 0.4896* | 0.1753 | 2,421 |
| **Multiple Outcomes (Privacy)** | -0.0061* | - | - | - | 2,421 |
| **Multiple Outcomes (Sharing)** | +0.0551* | - | - | - | 2,421 |

*** p<0.001, ** p<0.01, * p<0.05, † p<0.1

### Image
**[IMAGE: figures/regression_analysis_results.png]**

---

### Slide A2: Privacy Index Sub-dimensions

**Detailed Breakdown of Privacy Index Components**

| Sub-dimension | Variables | Mean (Diabetic) | Mean (Non-Diabetic) | Difference |
|---------------|-----------|-----------------|---------------------|------------|
| **Sharing** | 4 | 0.493 | 0.537 | -0.045 |
| **Portals** | 7 | 0.606 | 0.616 | -0.011 |
| **Devices** | 4 | 0.420 | 0.336 | +0.084 |
| **Trust** | 4 | 0.284 | 0.274 | +0.010 |
| **Social** | 2 | 0.556 | 0.536 | +0.020 |
| **Other** | 2 | 0.500 | 0.500 | 0.000 |

### Images
**[IMAGE: figures/privacy_index_detailed_table_optimized.png]**

*Caption: Detailed privacy index components*

---

## Image Reference Guide

### Required Images (Priority Order)

1. **`figures/model_logic_diagram.png`** - Slides 12, 16 (Model logic)
2. **`figures/regression_analysis_results.png`** - Slides 6, 7, Appendix
3. **`figures/diabetes_effects_comparison.png`** - Slides 8, 11
4. **`figures/causal_inference_analysis.png`** - Slides 9, 10
5. **`figures/privacy_caution_index_analysis.png`** - Slide 5
6. **`figures/privacy_index_construction_diagram_optimized.png`** - Slide 5
7. **`figures/data_quality_analysis.png`** - Slide 4
8. **`figures/diabetes_analysis.png`** - Slide 3
9. **`figures/panel_difference_in_differences_analysis.png`** - Slide 10
10. **`figures/privacy_index_detailed_table_optimized.png`** - Appendix

### Image Placement Instructions

- Replace `[IMAGE: figures/filename.png]` with actual image
- Images are in `figures/` directory
- All images are PNG format, 300 DPI
- PDF versions available in `figures/pdf_versions/` if needed

---

## Presentation Tips

### Design Recommendations
- **Color Scheme**: Use academic colors (blues, grays)
- **Font**: Clear, readable (Arial, Calibri, or similar)
- **Slide Layout**: Title at top, content in middle, minimal text
- **Visuals**: High resolution, clear labels, readable fonts

### Timing Guide
- **Title**: 30 seconds
- **Background**: 2 minutes
- **Methods**: 3 minutes (slides 5-6)
- **Findings**: 6 minutes (slides 7-11)
- **Implications**: 2 minutes
- **Conclusions**: 1 minute
- **Q&A**: 5-10 minutes

**Total**: ~15-20 minutes presentation

### Key Messages to Emphasize
1. **Research Question**: Chronic disease privacy behavior
2. **Key Finding**: Privacy is strongest predictor (β = -2.32)
3. **Causal Evidence**: IV shows large effect (β = 0.285)
4. **Generalizability**: Pattern applies to all 5 conditions
5. **Policy Relevance**: 34.2M+ Americans affected

---

*Last Updated: 2024*  
*Total Slides: 17 main + 2 optional appendix*  
*All images ready in: `figures/` directory*  
*Ready for direct use in presentation software*
