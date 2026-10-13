<p align="center">
  <img
    src="https://capsule-render.vercel.app/api?type=waving&amp;color=0:081A2D,50:0E7490,100:38BDF8&amp;height=230&amp;section=header&amp;text=Credit%20Behavior%20Intelligence&amp;fontSize=42&amp;fontColor=FFFFFF&amp;animation=fadeIn&amp;fontAlignY=35&amp;desc=PCA%20%7C%20K-Means%20%7C%20DBSCAN%20%7C%20Gradient%20Boosting&amp;descSize=18&amp;descAlignY=55"
    alt="Credit Behavior Intelligence project banner"
    width="100%"
  />
</p>

<h1 align="center">Credit Behavior Intelligence</h1>

<p align="center">
  An end-to-end machine-learning study of credit behavior using dimensionality reduction, clustering, anomaly exploration, and calibrated classification.
</p>

<p align="center">
  <a href="notebooks/credit-behavior-intelligence.ipynb"><img src="https://img.shields.io/badge/Notebook-Kaggle-20BEFF?style=for-the-badge&amp;logo=kaggle&amp;logoColor=white" alt="Kaggle notebook" /></a>
  <a href="source/credit_behavior_intelligence"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python 3.11 or newer" /></a>
  <a href="source/tests"><img src="https://img.shields.io/badge/Tests-16%20Passing-2EA44F?style=for-the-badge" alt="16 tests passing" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-7C3AED?style=for-the-badge" alt="MIT License" /></a>
</p>

<p align="center">
  <strong>PCA</strong> | <strong>K-Means</strong> | <strong>DBSCAN</strong> | <strong>Histogram Gradient Boosting</strong> | <strong>Probability Calibration</strong>
</p>

---

Credit Behavior Intelligence is a machine-learning portfolio project that studies historical credit-card customer behavior. It combines dimensionality reduction, customer segmentation, anomaly exploration, and historical default prediction in one leakage-aware workflow.

The project is designed to show how a complete machine-learning experiment should be organized, evaluated, explained, and presented. It is an educational analysis and must not be used to approve or reject real credit applications.

## Project files

- [Executed Kaggle notebook](notebooks/credit-behavior-intelligence.ipynb)
- [Presentation PDF](presentation/Credit_Behavior_Intelligence.pdf)
- [Editable PowerPoint presentation](presentation/Credit_Behavior_Intelligence.pptx)
- [Reusable Python source package](source/credit_behavior_intelligence)
- [Source-code tests](source/tests)

## The problem in simple language

The dataset contains information about credit limits, monthly bills, previous payments, and repayment history. The project asks four different questions:

1. Can many related variables be summarized with fewer dimensions?
2. Are there broad groups of customers with similar behavior?
3. Are some records unusually far from the main behavioral pattern?
4. Can historical behavior help estimate whether a customer belongs to the historical default class?

These questions do not all use the same type of machine learning.

## Where supervised and unsupervised learning are used

### Unsupervised learning

Unsupervised learning analyzes features without using a correct answer or target label during fitting.

The unsupervised part of this project includes:

| Technique | Where it is used | What it does |
|---|---|---|
| PCA | Dimensionality-reduction stage | Compresses 19 related behavioral variables into fewer components |
| K-Means | Customer-segmentation stage | Divides records into broad groups based on behavioral similarity |
| DBSCAN | Anomaly-exploration stage | Finds a dense main group and records located in sparse areas |

The historical default target is not used to fit PCA, K-Means, or DBSCAN. This is important because using the target would turn the analysis into a supervised task and could create misleading clusters.

### Supervised learning

Supervised learning trains a model using both input features and a known target label.

The supervised part uses a `HistGradientBoostingClassifier` to estimate the probability of historical default. The target contains:

- `0`: no default in the following month
- `1`: default in the following month

The model learns relationships between engineered behavioral features and these known historical outcomes.

## Dataset

The project uses the UCI Default of Credit Card Clients dataset, dataset ID 350.

| Property | Value |
|---|---:|
| Customer records | 30,000 |
| Raw explanatory features | 23 |
| Historical defaults | 6,636 |
| Historical default rate | 22.12% |
| Missing values | 0 |
| Dataset license | CC BY 4.0 |

A feature is an input variable used to describe a record. Examples include credit limit, bill amount, payment amount, and repayment status.

A target is the outcome the supervised model tries to estimate. In this project, the target is historical default in the following month.

Class imbalance means the target classes do not contain equal numbers of records. Only 22.12% of the records belong to the default class, so accuracy alone can hide weak performance on that class.

## Step 1: Load and validate the data

The notebook retrieves the official dataset with `ucimlrepo` and verifies:

- The expected 23 feature columns are present
- The target contains only `0` and `1`
- No feature or target values are missing
- Customer identifiers are unique
- Features contain numeric values
- Feature, target, and identifier row counts match

Validation means checking that data follows the expected structure before using it for analysis. It prevents silent mistakes such as missing columns, invalid labels, or misaligned records.

The customer identifier is never used as a model feature. An identifier distinguishes records but does not describe customer behavior.

## Step 2: Create the 80/10/10 split

The data is shuffled and divided into three partitions:

| Partition | Records | Purpose |
|---|---:|---|
| Training | 24,000 | Fit preprocessing and compare model candidates |
| Validation | 3,000 | Select the final probability threshold |
| Test | 3,000 | Perform one final evaluation |

A training set is the data used to learn model parameters.

A validation set is separate data used to make development decisions, such as selecting a probability threshold.

A test set is untouched data used only after all model decisions have been made. It estimates how the finished workflow performs on unseen records.

Stratification means preserving approximately the same target-class percentage in every partition.

Data leakage happens when information from validation or test data influences training. Leakage produces unrealistically strong results. This project fits transformations on training data only, selects the threshold on validation data, and opens the test set once at the end.

The split audit found:

- Zero identifier overlap among partitions
- Maximum absolute standardized mean difference of 0.0742
- Ten test rows sharing an exact feature profile with a development row
- Twenty-one exact feature profiles with conflicting target labels

A standardized mean difference measures how far apart two feature averages are after accounting for variation. Smaller values indicate more similar partitions.

Conflicting labels mean that identical recorded feature profiles sometimes have different outcomes. This is direct evidence that the available variables cannot determine the target perfectly.

## Step 3: Preprocess behavioral variables

The unsupervised workflow uses 19 behavioral variables:

- One credit-limit variable
- Six monthly bill variables
- Six monthly payment variables
- Six monthly repayment-status variables

Preprocessing means transforming raw variables into a form that an algorithm can use reliably.

Scaling places variables on comparable numerical ranges. Without scaling, a large bill amount could dominate a small repayment-status code simply because its number is larger.

Standard scaling subtracts a variable's average and divides by its standard deviation. The result is centered near zero with a comparable spread.

Robust scaling uses the median and interquartile range. It is less influenced by extreme values.

The Yeo-Johnson transformation reduces strong skew while supporting positive, zero, and negative values. This matters because some bill balances are negative.

The selected mixed preprocessing strategy applies a power transformation to monetary variables and standard scaling to repayment-status variables.

## Step 4: PCA dimensionality reduction

PCA means Principal Component Analysis. It is an unsupervised technique that combines correlated variables into new variables called principal components.

A principal component is a weighted combination of the original features. The first component captures the largest available direction of variation. Each later component captures a new direction that is not already represented by earlier components.

Explained variance is the percentage of data variation represented by a component.

Cumulative explained variance is the total variation represented by the first several components together.

Project results:

- Behavioral inputs: 19
- Components retained: 13
- Cumulative explained variance: 95.63%
- Variance explained by the first two components: 59.29%

The first two components are useful for two-dimensional visualization, but they do not contain all information in the dataset. The retained PCA representation is used for unsupervised analysis. The supervised classifier uses engineered behavioral features instead of PCA components.

## Step 5: K-Means customer segmentation

K-Means is an unsupervised clustering algorithm. Clustering means grouping records that are similar to one another without using known target labels.

The value `K` is the number of clusters requested from the algorithm.

A centroid is the numerical center of a cluster. K-Means repeatedly assigns each record to its nearest centroid and then updates the centroids.

The project compares several cluster counts and PCA representations using the following measurements:

- Inertia: the total squared distance between records and their assigned centroids. Lower inertia means records are closer to their centers, but inertia always decreases when more clusters are added.
- Silhouette score: compares how close a record is to its own cluster versus other clusters. Values closer to `1` indicate clearer separation, values near `0` indicate overlap, and negative values suggest possible incorrect assignments.
- Calinski-Harabasz score: compares separation between clusters with compactness inside clusters. Larger values are generally better.
- Davies-Bouldin score: measures similarity between each cluster and its closest competing cluster. Lower values are generally better.
- Minimum cluster share: the percentage of records in the smallest cluster. It prevents selecting a solution built around a tiny, unstable group.
- Adjusted Rand index: measures agreement between two cluster assignments while correcting for agreement expected by chance. A value near `1` indicates highly stable assignments.

Selected K-Means result:

| Measurement | Result |
|---|---:|
| PCA components | 10 |
| Clusters | 2 |
| Silhouette score | 0.2638 |
| Mean stability adjusted Rand index | 0.9960 |
| Cluster shares | 75.34% and 24.66% |

The high stability score means repeated initializations produce almost the same assignments. The moderate silhouette score means the groups still overlap. Therefore, they are broad behavioral segments, not perfect customer types and not credit-risk classes.

## Step 6: DBSCAN anomaly exploration

DBSCAN means Density-Based Spatial Clustering of Applications with Noise. It is an unsupervised algorithm that looks for dense groups separated by sparse regions.

Density describes how closely records are packed together.

`eps` is the maximum neighborhood distance used when deciding whether records are close enough to support one another.

`min_samples` is the minimum number of nearby records required to form a dense region.

A core point has enough neighbors to belong to a dense region.

A noise point does not have enough nearby support under the selected settings. Noise means unusual location in feature space. It does not mean fraud, misconduct, or incorrect data.

Whitened PCA rescales retained components so that each has unit variance. This prevents the highest-variance component from dominating distance calculations.

Selected DBSCAN result:

| Setting or result | Value |
|---|---:|
| Whitened PCA components | 5 |
| `eps` | 1.1526 |
| `min_samples` | 10 |
| Substantial non-noise groups | 1 |
| Noise share | 2.08% |

DBSCAN did not find several credible customer segments. It found one main dense group and a small number of sparse records. The honest interpretation is anomaly exploration rather than customer segmentation.

## Step 7: Engineer supervised-learning features

Feature engineering means creating useful model inputs from raw variables using domain-informed calculations.

The supervised pipeline creates 54 features from 19 non-demographic behavioral inputs. Examples include:

- Bill-to-limit ratios
- Payment-to-limit ratios
- Payment-to-bill ratios
- Average and variation of bills
- Average and total payments
- Recent bill and repayment changes
- Number of delayed-payment months
- Number of serious-delay months

A ratio compares one quantity with another. For example, bill-to-limit ratio describes how large a bill is relative to the available credit limit.

Demographic fields are excluded from prediction. They remain available only for aggregate subgroup auditing.

## Step 8: Train the supervised classifier

The supervised model is `HistGradientBoostingClassifier`.

A classifier is a model that estimates which class a record belongs to.

A decision tree makes predictions by dividing records through a sequence of feature-based rules.

Gradient boosting builds trees one after another. Each new tree focuses on reducing errors left by the existing collection of trees.

Histogram gradient boosting groups continuous feature values into bins before finding tree splits. This can make training faster and more memory efficient.

The project compares ten candidate configurations with three-fold stratified cross-validation.

A candidate configuration is one combination of model settings.

Cross-validation repeatedly divides the training data into fitting and evaluation folds. It estimates whether a configuration performs consistently instead of depending on one temporary split.

A fold is one subset used during cross-validation.

The primary candidate-selection measurement is PR-AUC because the positive class is less common.

Selected model settings:

| Setting | Value | Meaning |
|---|---:|---|
| Learning rate | 0.05 | Controls how strongly each new tree changes the model |
| Maximum iterations | 600 | Upper limit on the number of boosting rounds |
| Maximum leaf nodes | 15 | Limits the complexity of each tree |
| Minimum samples per leaf | 20 | Prevents leaves based on very few records |
| L2 regularization | 1.0 | Penalizes overly large model updates |
| Early-stopping patience | 30 | Stops after 30 rounds without meaningful validation improvement |

A boosting iteration is one round in which another tree is added. It is similar to an epoch in the sense that it is a repeated training step, but it is not a neural-network epoch.

A leaf node is a final region of a decision tree that produces a prediction contribution.

Regularization discourages unnecessary model complexity and helps reduce overfitting.

## Learning rate, early stopping, and loss

The learning rate controls how much each new tree contributes. A small learning rate makes slower, more conservative updates. A large learning rate learns faster but may overshoot useful patterns or overfit.

The project compared fixed learning rates of `0.01`, `0.03`, `0.05`, and `0.08`. Cross-validation selected `0.05`.

A learning-rate decay scheduler gradually reduces the learning rate during training. This is common in neural networks and is available through callbacks in libraries such as XGBoost and LightGBM.

Scikit-learn's `HistGradientBoostingClassifier` does not provide a native reduce-on-plateau learning-rate scheduler. It uses a fixed learning rate, so this project compares fixed values through cross-validation and uses early stopping.

Early stopping monitors performance on internal validation data and stops training when additional iterations no longer improve it.

Patience is the number of non-improving rounds allowed before training stops.

Loss is a numerical measurement of prediction error used during training. Lower loss means the predicted probabilities better match the observed outcomes.

Log loss strongly penalizes confident probability predictions that are wrong. It evaluates the quality of probabilities rather than only checking final class labels.

The best internal-validation iterations were 81, 72, and 138. The final calibrated estimators stopped at 116, 131, and 106 iterations. No fold reached the 600-iteration cap.

Overfitting happens when training performance keeps improving while validation performance stops improving or becomes worse. The model has learned details that do not generalize well.

Underfitting happens when a model is too simple or insufficiently trained to learn important patterns in both training and validation data.

## Step 9: Calibrate probabilities

Probability calibration checks whether predicted probabilities match observed frequencies.

For example, among many records receiving a calibrated probability near `0.40`, approximately 40% should belong to the positive class.

The project uses sigmoid calibration across three folds.

Sigmoid calibration learns a smooth S-shaped correction that maps raw model scores to better-aligned probabilities.

The Brier score and calibration curve are used to evaluate probability quality.

## Step 10: Select the decision threshold

A probability threshold converts a probability into a class prediction.

At the default threshold of `0.50`, probabilities at or above 0.50 become positive predictions.

The project selects a threshold of `0.2916` using validation data to improve F1 score. Lowering the threshold identifies more historical defaults but also creates more false-positive predictions.

Threshold selection is an operating decision. Different applications may prefer different balances between missed positives and false alarms.

## Step 11: Evaluate the final test set

A confusion matrix counts the four possible prediction outcomes:

- True negative: predicted no default, and the actual class is no default
- False positive: predicted default, but the actual class is no default
- False negative: predicted no default, but the actual class is default
- True positive: predicted default, and the actual class is default

At threshold `0.2916`, the confusion matrix contains:

| Outcome | Count |
|---|---:|
| True negatives | 2,055 |
| False positives | 281 |
| False negatives | 305 |
| True positives | 359 |

### Evaluation metrics explained

Accuracy is the percentage of all predictions that are correct. It can be misleading when one class is more common.

Balanced accuracy calculates recall separately for each class and averages the results. It gives equal importance to the default and no-default classes.

Precision answers: among the records predicted as default, what percentage actually defaulted? Higher precision means fewer false alarms.

Recall answers: among all records that actually defaulted, what percentage did the model identify? Higher recall means fewer missed defaults. Recall is also called sensitivity or true-positive rate.

F1 score is the harmonic mean of precision and recall. It becomes high only when both precision and recall are reasonably strong.

Specificity answers: among records that did not default, what percentage did the model correctly identify as no default? It is also called true-negative rate.

ROC-AUC means Area Under the Receiver Operating Characteristic Curve. It measures how well the model ranks positive records above negative records across all possible thresholds. A value of `0.50` is similar to random ranking, while `1.00` is perfect ranking.

PR-AUC means Area Under the Precision-Recall Curve. It summarizes the tradeoff between precision and recall across thresholds. It is particularly informative when the positive class is less common.

Brier score is the average squared difference between predicted probabilities and actual binary outcomes. Lower values indicate better probability accuracy.

Log loss evaluates how closely predicted probabilities match actual outcomes and strongly penalizes confident mistakes. Lower values are better.

### Final results

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

ROC-AUC, PR-AUC, Brier score, and log loss do not change between the two rows because the underlying probabilities are unchanged. Only the threshold used to create class predictions changes.

## Why the project does not claim 90% F1

The feature distributions overlap, the target is imbalanced, some categories are rare, and 21 identical feature profiles have conflicting outcomes. The available variables therefore do not determine the target perfectly.

Reaching 90% F1 would probably require substantially different data, a different target, or leakage. Reporting the measured test result is more credible than tuning repeatedly against the test set.

## Source-code map

The notebook remains self-contained so it can run directly in Kaggle. The `source/` folder provides the same workflow as reusable Python modules:

| Module | Responsibility |
|---|---|
| `config.py` | Reproducible settings and selected model configurations |
| `data.py` | UCI loading, column normalization, and dataset contract |
| `validation.py` | Data-quality, category-code, and conflicting-label audits |
| `splitting.py` | Stratified train, validation, and test splitting |
| `preprocessing.py` | Scaling and Yeo-Johnson preprocessing |
| `pca.py` | PCA fitting, transformation, and variance summaries |
| `kmeans.py` | K-Means fitting, metrics, and stability checks |
| `dbscan.py` | Neighbor distances, DBSCAN fitting, and density summaries |
| `features.py` | The 54-feature behavioral engineering contract |
| `modeling.py` | Gradient-boosting construction and candidate comparison |
| `calibration.py` | Sigmoid calibration and F1 threshold selection |
| `evaluation.py` | Classification metrics, subgroup summaries, and latency |
| `plotting.py` | Reusable model and evaluation plots |
| `artifacts.py` | Versioned model saving and loading |

## Repository structure

```text
credit-behavior-intelligence/
|-- notebooks/
|   `-- credit-behavior-intelligence.ipynb
|-- presentation/
|   |-- Credit_Behavior_Intelligence.pdf
|   `-- Credit_Behavior_Intelligence.pptx
|-- source/
|   |-- credit_behavior_intelligence/
|   |-- tests/
|   `-- requirements.txt
|-- README.md
|-- LICENSE
`-- .gitignore
```

## Run the project

Create an environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r source\requirements.txt
```

Run the source-code tests:

```powershell
python -m unittest discover -s source\tests -v
```

For the complete experiment, upload [credit-behavior-intelligence.ipynb](notebooks/credit-behavior-intelligence.ipynb) to Kaggle, enable Internet access, and run every cell in order. A GPU session can be enabled, but the scikit-learn models execute on the CPU.

## Responsible-use limitations

- This is an educational portfolio project, not a production credit-decision system.
- Clusters describe broad patterns and are not individual risk categories.
- DBSCAN noise indicates sparse behavior and is not evidence of fraud.
- Historical association does not establish causation.
- Demographic variables are excluded from predictive inputs.
- Real deployment would require legal review, fairness analysis, security, monitoring, current representative data, and documented human oversight.

## License and attribution

Project code is released under the [MIT License](LICENSE). The dataset remains subject to the UCI attribution requirements and the Creative Commons Attribution 4.0 license.
