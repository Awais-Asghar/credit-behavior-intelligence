# Customer Intelligence Presentation Source

This document is the grounded source for generating a presentation in NotebookLM or Gemini. It contains the verified story, numbers, visuals, limitations, and presentation instructions for the project.

## Ready-to-paste generation prompt

Create a 12-slide, 16:9 presentation for a machine-learning job interview using only the supplied Customer Intelligence project sources.

Requirements:

- Use a clean professional visual style with deep navy, blue, cyan, violet, and restrained orange accents.
- Keep each slide focused on one message with three to five concise bullets.
- Use the supplied project figures instead of inventing charts.
- Include short speaker notes that explain the technical decision behind each slide.
- Preserve every metric exactly as written in the source.
- Clearly separate dimensionality reduction, segmentation, anomaly exploration, and supervised prediction.
- Do not describe K-Means clusters as risk classes.
- Do not describe DBSCAN noise as fraud.
- Do not claim that the model should approve or reject real credit applications.
- Explain why 90% accuracy and 90% F1 are not credible goals for this dataset without leakage or a different problem definition.
- Finish with lessons learned, limitations, and the repository link.

## One-sentence project description

Customer Intelligence is a leakage-aware machine-learning portfolio project that combines PCA, K-Means, DBSCAN, and calibrated histogram gradient boosting to study historical credit-card customer behavior.

## Audience and presentation goal

The audience is an ML recruiter, interviewer, or engineering manager. The goal is to demonstrate:

- Correct separation of supervised and unsupervised objectives
- Careful train, validation, and test design
- Feature engineering and model selection
- Threshold selection and probability calibration
- Early stopping and loss-curve interpretation
- Honest evaluation on noisy, imbalanced data
- Responsible interpretation of customer and credit-related results

## Verified project facts

### Dataset

- Source: UCI Default of Credit Card Clients, dataset ID 350
- Records: 30,000
- Raw explanatory features: 23
- Historical defaults: 6,636, or 22.12%
- Missing values: none in the validated dataset
- Dataset license: CC BY 4.0

### Split and leakage controls

- Training: 24,000 records, or 80%
- Validation: 3,000 records, or 10%
- Test: 3,000 records, or 10%
- Shuffled and target-stratified with random state 42
- Zero identifier overlap across partitions
- Maximum pairwise absolute standardized mean difference: 0.0742
- Ten test rows share an exact 23-feature profile with a development row, but identifiers differ
- Twenty-one exact feature profiles have conflicting target labels
- The test set was not used for candidate or threshold selection

### PCA

- Behavioral inputs: 19
- Components retained: 13
- Retained cumulative variance: 95.63%
- Variance explained by the first two components: 59.29%
- The two-dimensional projection is a visualization, not the full modeling representation

### K-Means

- Selected representation: 10 PCA components
- Selected cluster count: 2
- Silhouette score: 0.2638
- Mean initialization-stability adjusted Rand index: 0.9960
- Cluster shares: 75.34% and 24.66%
- Interpretation: stable but overlapping behavioral segments

### DBSCAN

- Final purpose: anomaly exploration
- Selected representation: 5 whitened PCA components
- `eps`: 1.1526
- `min_samples`: 10
- Substantial non-noise groups: 1
- Noise share: 2.08%
- Interpretation: unusual multivariate behavior, not fraud or wrongdoing

### Supervised model

- Model: `HistGradientBoostingClassifier`
- Engineered behavioral features: 54 from 19 non-demographic inputs
- Candidate configurations: 10
- Selection: three-fold stratified cross-validation
- Primary candidate-selection metric: PR-AUC
- Probability calibration: sigmoid, three folds
- Selected learning rate: 0.05
- Maximum leaf nodes: 15
- Maximum boosting iterations: 600
- Early-stopping patience: 30
- Best internal validation iterations: 81, 72, and 138
- Final calibrated estimator iterations: 116, 131, and 106
- Validation-selected threshold: 0.2916

### Final test metrics

| Metric | Threshold 0.50 | Selected threshold 0.2916 |
|---|---:|---:|
| Accuracy | 82.47% | 80.47% |
| Balanced accuracy | 66.05% | 71.02% |
| Precision | 69.83% | 56.09% |
| Recall | 36.60% | 54.07% |
| F1 score | 48.02% | 55.06% |
| Specificity | 95.51% | 87.97% |
| ROC-AUC | 78.77% | 78.77% |
| PR-AUC | 57.13% | 57.13% |
| Brier score | 0.1323 | 0.1323 |
| Log loss | 0.4230 | 0.4230 |

Selected-threshold confusion matrix:

- True negatives: 2,055
- False positives: 281
- False negatives: 305
- True positives: 359

## Recommended slide sequence

### Slide 1: Customer Intelligence

Main message: One project combines complementary ML methods without mixing their purposes.

Include:

- PCA for compression and visualization
- K-Means for behavioral segmentation
- DBSCAN for anomaly exploration
- Calibrated gradient boosting for default-risk estimation

Visual: `docs/assets/customer-intelligence-banner.png`

Speaker note: Open by explaining that the project is designed to demonstrate complete ML reasoning, not only model training.

### Slide 2: Real-world problem

Main message: Customer behavior contains segments, anomalies, and uncertain outcomes.

Include:

- Segmentation supports descriptive analysis
- Anomaly exploration finds sparse behavior patterns
- Supervised learning estimates historical risk
- Each output requires a different interpretation

Speaker note: Stress that clustering outputs are not labels and anomalies are not evidence of misconduct.

### Slide 3: Dataset and class imbalance

Main message: The target is imbalanced but not extremely rare.

Include:

- 30,000 customers and 23 raw explanatory features
- 6,636 defaults, or 22.12%
- No missing values after validation
- UCI source with CC BY 4.0 licensing

Visual: `reports/figures/class-distribution.png`

Speaker note: Explain why accuracy alone can hide weak minority-class performance.

### Slide 4: Leakage-safe experimental design

Main message: Model development and final evaluation are separated.

Include:

- 80% training, 10% validation, 10% test
- Target stratification and shuffling
- Zero identifier overlap
- Candidate selection on training folds
- Threshold selection on validation only

Speaker note: State that the test set is opened once after the model and threshold are frozen.

### Slide 5: PCA findings

Main message: Most information is retained with fewer dimensions.

Include:

- 19 behavioral variables
- 13 components retain 95.63% variance
- First two components retain 59.29%
- Two dimensions support visualization but not full inference

Visuals:

- `reports/figures/pca-preprocessing-comparison.png`
- `reports/figures/pca-diagnostic-projection.png`

Speaker note: Mention that preprocessing was fitted on training data to prevent leakage.

### Slide 6: K-Means segmentation

Main message: Two clusters are reproducible but overlap.

Include:

- Ten-component PCA representation
- Two selected clusters
- Silhouette score 0.2638
- Stability ARI 0.9960
- Cluster shares 75.34% and 24.66%

Visuals:

- `reports/figures/kmeans-model-selection.png`
- `reports/figures/kmeans-cluster-projection.png`

Speaker note: Contrast strong initialization stability with only moderate geometric separation.

### Slide 7: DBSCAN result

Main message: The data supports anomaly exploration more than density-based segmentation.

Include:

- Five whitened PCA components
- One substantial core group
- 2.08% noise records
- No credible multiple-cluster density solution

Visuals:

- `reports/figures/dbscan-distance-curve.png`
- `reports/figures/dbscan-anomaly-projection.png`

Speaker note: Present the negative result as good scientific judgment instead of forcing an attractive conclusion.

### Slide 8: Feature engineering and candidate selection

Main message: The supervised pipeline uses behavior-derived features and cross-validation.

Include:

- 54 engineered features from 19 behavioral inputs
- Demographics excluded from prediction
- Ten histogram-gradient-boosting candidates
- Three-fold stratified cross-validation
- PR-AUC used as the primary selection metric

Speaker note: Explain that demographics remain available only for aggregate subgroup auditing.

### Slide 9: Early stopping and generalization

Main message: Training was stopped when internal validation loss plateaued.

Include:

- Maximum 600 boosting iterations
- Early-stopping patience 30
- Best iterations 81, 72, and 138
- No fold reached the iteration cap
- Third fold shows the largest train-validation gap

Visual: `reports/figures/training-validation-loss.png`

Speaker note: Explain why increasing the epoch or iteration limit would not improve the observed run.

### Slide 10: Threshold selection and calibration

Main message: The classification threshold is an operating decision, not a fixed law.

Include:

- Sigmoid probability calibration
- Validation-selected threshold 0.2916
- Selected threshold improves recall and F1
- Accuracy and precision decrease as the model flags more positives

Visual: `reports/figures/threshold-and-calibration.png`

Speaker note: Explain that ROC-AUC and PR-AUC do not change when only the decision threshold changes.

### Slide 11: Final test performance

Main message: The final result is useful for learning and honest about its limits.

Include:

- Accuracy 80.47%
- Balanced accuracy 71.02%
- Precision 56.09%
- Recall 54.07%
- F1 55.06%
- ROC-AUC 78.77% and PR-AUC 57.13%

Visual: `reports/figures/test-confusion-matrix-and-roc.png`

Speaker note: Compare the selected threshold with 0.50 and explain the false-positive versus false-negative tradeoff.

### Slide 12: Lessons, limits, and next steps

Main message: Strong ML work includes knowing what the data cannot support.

Include:

- Conflicting labels prevent deterministic prediction
- Rare categories increase uncertainty
- Clusters are descriptive, not causal
- DBSCAN noise is not fraud
- Production use would require governance, monitoring, and jurisdiction-specific review
- Repository includes the executed notebook, tests, figures, and reproducibility instructions

Speaker note: Close by emphasizing leakage prevention, honest evaluation, and clear communication of uncertainty.

## Interview questions to prepare for

### Why did you choose PR-AUC for candidate selection?

The positive class represents 22.12% of records. PR-AUC focuses on performance for the positive class and is more informative than accuracy when class frequencies differ.

### Why is the selected threshold below 0.50?

The threshold was selected on validation data to maximize F1. Lowering it from 0.50 identifies more historical defaults, improving recall and balanced accuracy at the cost of precision and overall accuracy.

### Why did you not achieve 90% F1?

The behavior of the two classes overlaps, and 21 identical feature profiles have conflicting labels. The available inputs therefore cannot determine the outcome perfectly. A 90% result would require materially different data, a different target, or likely leakage.

### Why use PCA before K-Means?

The input variables are correlated and have different distributions. PCA provides a compact representation that can reduce redundancy and make distance-based clustering more stable and interpretable.

### Why keep DBSCAN if it did not produce useful clusters?

It answered a different question. DBSCAN showed that the tested density structure consists mainly of one core group plus sparse records. That supports anomaly exploration and prevents a misleading claim about natural customer segments.

### Is the model ready for real credit decisions?

No. It is an educational portfolio model trained on historical observational data. Real use would require legal and policy review, stronger fairness analysis, monitoring, security, human oversight, and validation on current representative data.

## Source files to provide to NotebookLM or Gemini

Provide these files together:

1. `README.md`
2. `docs/presentation-source.md`
3. `ml/notebooks/customer-intelligence.ipynb`
4. The images under `reports/figures/`

Use the README and this document for the narrative. Use the notebook as the detailed evidence source. Use only the exported figures for charts.
