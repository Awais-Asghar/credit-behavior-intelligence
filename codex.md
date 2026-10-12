# Customer Intelligence ML Project

Last updated: 2026-10-12

## Current scope

The current deliverable is a public GitHub portfolio repository, not a deployed dashboard. The repository contains the executed Kaggle notebook, reusable validation and feature-engineering code, unit tests, exported result figures, a visual README, and a NotebookLM or Gemini presentation source.

The earlier Next.js, FastAPI, and Vercel sections below are retained as historical planning context only. They are not part of the active project scope unless the user explicitly restores that direction.

The canonical Kaggle run uses an 80/10/10 split and selects `hgb_lr0.05_leaf15`. Its validation-selected threshold is 0.2916. Final test performance is 80.47% accuracy, 71.02% balanced accuracy, 56.09% precision, 54.07% recall, 55.06% F1, 78.77% ROC-AUC, and 57.13% PR-AUC.

## Purpose of this file

This file is the living context document for the project. Update it whenever the project scope, dataset, architecture, modeling decisions, API contract, evaluation strategy, or deployment approach changes. Future work should read this file before making project changes.

## Project summary

Build a portfolio-quality customer intelligence application that demonstrates four machine learning techniques:

1. Principal Component Analysis for dimensionality reduction and visualization.
2. K-Means for behavioral customer segmentation.
3. DBSCAN for density-based structure discovery and anomaly exploration.
4. Feature-engineered histogram gradient boosting for default-risk classification.

Histogram gradient boosting is the only supervised classifier retained in the current notebook and is the provisional deployment model.

The final product is a reproducible GitHub portfolio project with an executed notebook, documented results, exported figures, tests, and presentation material. Model training happens offline in Kaggle or a compatible local environment.

This is an educational decision-support demonstration. It must not be presented as a system that should approve or reject real credit applications.

## Working project name

**Customer Intelligence: Segmentation, Anomaly Detection, and Default Risk Modeling**

Local folder name: `Customer Intelligence`

Repository slug: `customer-intelligence-ml`

## Intended audience

- Machine learning recruiters and hiring managers
- Interviewers evaluating practical ML understanding
- Engineers reviewing reproducibility and code quality
- Learners exploring supervised and unsupervised learning

## Dataset decision

### Selected dataset

**Default of Credit Card Clients** from the UCI Machine Learning Repository.

- Official page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- DOI: https://doi.org/10.24432/C55S3H
- Creator: I-Cheng Yeh
- Instances: 30,000
- Explanatory features: 23
- Target: Default payment in the following month, represented as a binary label
- Missing values reported by UCI: None
- License: Creative Commons Attribution 4.0 International, CC BY 4.0
- Original context: Credit card clients in Taiwan, with payment data from 2005

### Main feature groups

- Credit limit
- Repayment status across six months
- Bill statement amounts across six months
- Previous payment amounts across six months
- Age, education, marital status, and sex demographic fields
- Customer identifier, which must not be used as a model feature

### Why this dataset fits the project

- It contains a real target for supervised default-risk classification.
- Its financial behavior features support meaningful customer segmentation.
- Its correlated monthly variables provide a clear reason to study PCA.
- Its size is large enough to demonstrate realistic preprocessing and evaluation.
- DBSCAN can be applied in a reduced PCA space to explore dense groups and unusual records.
- Its license allows sharing and adaptation with attribution.

### Confirmed observations from Kaggle

- Notebook 01 completed successfully in Kaggle.
- The dataset contains 30,000 rows and 23 explanatory features.
- There are no missing values and no duplicate identifiers.
- The target contains 23,364 class 0 records and 6,636 class 1 records, giving a 22.12% historical default rate.
- Education contains observed codes from 0 through 6, while the UCI page documents codes 1 through 4.
- Marriage contains observed code 0, while the UCI page documents codes 1 through 3.
- Repayment-status fields contain codes from -2 through 8. The UCI page does not fully document the meanings of -2 and 0.
- Bill-statement columns contain negative values and large positive extremes. These values will be preserved until their semantics and modeling impact are analyzed.

### Data acquisition

Preferred reproducible loading method:

```python
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=350)
X = dataset.data.features
y = dataset.data.targets
```

The Python dependency is `ucimlrepo`. The original `.xls` file is also available from the official UCI page. Kaggle is the selected notebook and offline model-training environment. Kaggle Internet access must be enabled when fetching the dataset directly from UCI. A later data-ingestion task will cache a versioned local copy and record a checksum. Raw data has not been downloaded into the local project yet.

## Business framing

The application answers four related questions:

1. Which broad behavioral patterns explain most of the variation among customers?
2. Which customer behavior segments appear in the data?
3. Which records have unusual repayment or billing patterns?
4. Can gradient boosting estimate whether a record belongs to the historical default class?

The dashboard should translate model outputs into clear observations while distinguishing association from causation.

## Machine learning scope

### PCA

- Fit PCA only on transformed training data.
- Plot cumulative explained variance and the first two or three components.
- Compare downstream model performance and latency with and without PCA.
- Explain that PCA components are combinations of original features and may reduce interpretability.
- Compare standard scaling, robust scaling, and a mixed preprocessing strategy before downstream modeling.
- Use Yeo-Johnson for skewed monetary features in the provisional mixed strategy because it supports negative and zero values.
- Keep repayment-status features on an ordinal numeric scale and standardize them separately.

### K-Means

- Cluster behavioral and financial features without using the target.
- Choose the number of clusters using silhouette score, elbow analysis, stability, and usefulness of the resulting profiles.
- Report silhouette, Calinski-Harabasz, and Davies-Bouldin metrics.
- Give clusters descriptive names only after examining their feature profiles.
- Compare the full preprocessed space with PCA representations retaining at least 90% and 95% of training variance.
- Use a fixed silhouette sample to avoid an unnecessary full pairwise-distance calculation.
- Reject automatic selections with a cluster smaller than 5% when viable alternatives exist.
- Check initialization stability with adjusted Rand index before treating a solution as reproducible.

### DBSCAN

- Run DBSCAN in a scaled, PCA-reduced feature space.
- Select `eps` using a nearest-neighbor distance plot and test several `min_samples` values.
- Use a fixed, target-independent sample of 8,000 training rows to keep the density search reproducible and memory-safe.
- Compare whitened PCA representations with 2, 5, and 10 components. Whitening prevents high-variance components from dominating Euclidean distance.
- Evaluate `min_samples` values of 10, 20, and 40 with `eps` candidates from the 90th, 95th, and 98th percentiles of k-nearest-neighbor distance.
- Require at least two non-noise clusters, a noise share between 1% and 30%, and a minimum cluster share of 2% before presenting DBSCAN as segmentation.
- If no candidate passes those checks, use DBSCAN only for anomaly exploration and state that result clearly.
- Treat the noise label as an unusual observation, not proof of fraud or wrongdoing.
- Report cluster count, noise rate, stability, and silhouette score when mathematically valid.
- The first dashboard version will visualize precomputed DBSCAN results. Scikit-learn DBSCAN does not provide a standard `predict()` method for new records.

### Supervised split and gradient-boosting classifier

- Use a fixed 80% training, 10% validation, and 10% test protocol with random state 42.
- Shuffle records and stratify both split operations on the target.
- Use 24,000 training rows for candidate selection and calibration, 3,000 validation rows for probability-threshold selection, and 3,000 test rows for final evaluation.
- Keep identifiers out of all model features and verify zero identifier overlap among all three partitions.
- Compare feature distributions across every pair of partitions using absolute standardized mean differences.
- Audit age bands, demographic codes, repayment-status codes, and demographic code-target combinations. Report rare missing combinations rather than claiming perfect coverage.
- Build a deterministic 54-feature behavioral representation from the 19 deployment inputs. The feature engineering includes utilization, payment-to-bill ratios, bill and payment summaries, trends, and delinquency counts.
- Compare ten histogram gradient-boosting configurations with three-fold stratified cross-validation on training data only. Test learning rates 0.01, 0.03, 0.05, and 0.08 with a maximum of 600 boosting iterations. Use PR-AUC as the primary selection metric.
- Use explicit internal early stopping with a 10% training-fold validation fraction, patience of 30 iterations, tolerance `1e-7`, and log loss scoring.
- Plot and export training and internal-validation loss curves for every calibration fold. Report the best validation-loss iteration, stopping iteration, and train-validation gap.
- Scikit-learn histogram gradient boosting uses gradient-based tree updates and no Adam or SGD optimizer. It has no native learning-rate decay callback, so fixed learning rates are compared through cross-validation.
- Calibrate the selected configuration on training data, select the operating threshold on validation data, then refit on the 27,000-row development partition before one final test evaluation.
- Save the selected calibrated estimator, feature contract, threshold, early-stopping settings, and actual iteration counts in the versioned joblib artifact.
- Keep both the default 0.50 threshold and validation-selected threshold in the report so accuracy does not hide missed defaults.

## Data and ethics rules

- Never use the customer identifier as a feature.
- Never use the classification target during clustering or PCA fitting.
- Split supervised data before fitting imputers, encoders, scalers, PCA, feature selection, or models.
- Start with behavioral and financial variables for deployed predictions and clusters.
- Keep demographic variables separate for subgroup evaluation and bias analysis unless a documented experiment requires them.
- Clearly state that the dataset is historical, geographically specific, and not representative of every population.
- Never describe a DBSCAN anomaly as confirmed fraud.
- Never recommend automatic approval, rejection, pricing, or adverse action based on this demonstration.
- Avoid exposing individual records in the public dashboard. Use aggregate views and synthetic example inputs.

## Planned application architecture

```text
Browser
  |
  v
Next.js dashboard on Vercel
  |
  v
FastAPI endpoints using Vercel Python Functions
  |
  v
Versioned preprocessing and model artifacts
```

### Frontend responsibilities

- Project overview and methodology
- Dataset explorer using aggregate statistics
- PCA explained-variance and component plots
- K-Means cluster visualization and profiles
- DBSCAN cluster and noise visualization
- Gradient-boosting evaluation, calibration, and loss-curve charts
- Synthetic customer input form for demonstration inference
- Limitations, ethics, and reproducibility sections

### Backend responsibilities

- Validate requests with Pydantic schemas
- Load versioned preprocessing and model artifacts
- Return K-Means segment assignments for valid synthetic inputs
- Return calibrated gradient-boosting predictions
- Serve precomputed PCA, clustering, anomaly, and evaluation summaries
- Expose model version and health information
- Never train models during a web request

### Candidate API routes

- `GET /api/health`
- `GET /api/model-info`
- `GET /api/analytics/summary`
- `POST /api/predict/default`
- `POST /api/predict/segment`

DBSCAN results will initially be exposed through analytics endpoints rather than a misleading prediction endpoint.

## Planned technology stack

### Experimentation and training

- Kaggle notebooks using CPU sessions by default
- `ml/notebooks/Customer Intelligence.ipynb` is the primary, self-contained Kaggle notebook.
- Numbered notebooks are retained as development references and must not be required to run the primary notebook.
- `ucimlrepo==0.0.7` for reproducible access to UCI dataset 350
- Offline preprocessing, model selection, evaluation, and artifact export
- Fixed random seeds where the algorithm supports them
- Notebook outputs reviewed before any artifact is used by the API

### Frontend

- Next.js
- TypeScript
- Tailwind CSS
- Plotly or Recharts for interactive charts
- Vitest and React Testing Library

### Backend and ML

- FastAPI
- Pydantic
- pandas and NumPy
- scikit-learn
- joblib
- pytest
- Ruff for linting and formatting

### Deployment

- Vercel for the Next.js application and Python functions
- Git-based preview deployments before production
- Environment variables managed through Vercel, never committed
- Compact, pre-trained model artifacts included with or retrieved by the deployment

Vercel's Python runtime is currently documented as Beta. Dependency size, cold-start latency, function duration, and model artifact size must be checked before the deployment architecture is finalized.

## Planned repository structure

```text
Customer Intelligence/
|-- codex.md
|-- README.md
|-- frontend/
|   |-- app/
|   |-- components/
|   |-- lib/
|   `-- tests/
|-- api/
|   `-- index.py
|-- ml/
|   |-- data/
|   |-- notebooks/
|   |-- src/
|   |-- tests/
|   `-- artifacts/
|-- requirements.txt
|-- package.json
`-- vercel.json
```

The exact Vercel-compatible layout will be validated with a minimal deployment before the full UI is implemented.

## Delivery phases

### Phase 1: Foundation

- Create repository structure and dependency files.
- Add reproducible dataset ingestion and validation.
- Perform exploratory data analysis.
- Define preprocessing and leakage-safe splits.

### Phase 2: Modeling

- Build PCA analysis.
- Build and interpret K-Means models.
- Build and interpret DBSCAN models.
- Train, tune, calibrate, and diagnose the gradient-boosting classifier.
- Save metrics, figures, and versioned artifacts.

### Phase 3: API

- Define request and response schemas.
- Implement health, metadata, analytics, segment, and default endpoints.
- Add unit and integration tests.
- Measure inference latency and artifact size.

### Phase 4: Dashboard

- Build the project narrative and dataset overview.
- Add interactive PCA and clustering visualizations.
- Add model evaluation pages.
- Add a synthetic inference demo.
- Add limitations and responsible-use explanations.

### Phase 5: Deployment and portfolio polish

- Validate a Vercel preview deployment.
- Improve accessibility, responsiveness, and error states.
- Add README setup instructions, screenshots, architecture, results, and attribution.
- Prepare a five-minute interview demo and a short backup recording.

## Definition of done

- Dataset loading is reproducible from the official source.
- Training and inference preprocessing are consistent and leakage-safe.
- PCA, K-Means, DBSCAN, and histogram gradient boosting are implemented and evaluated.
- Clustering results are interpreted without using the target to create clusters.
- The FastAPI contract is tested.
- The Next.js dashboard works on desktop and mobile.
- A Vercel preview deployment succeeds within platform limits.
- The README explains setup, methods, results, limitations, ethics, and dataset attribution.
- No credentials, tokens, private customer data, or generated secrets are committed.

## Current status

- Project folder created.
- Project context initialized in `codex.md`.
- Dataset selected and verified from UCI.
- Kaggle selected as the experimentation and offline training environment.
- Kaggle-ready data loading and validation notebook created.
- Notebook 01 executed successfully by the user in Kaggle.
- Notebook 02 executed successfully by the user in Kaggle without errors.
- Standalone exploratory data analysis notebook created with an early stratified holdout split.
- Standalone preprocessing and PCA notebook created with three preprocessing comparisons and provisional artifact export.
- Combined `Customer Intelligence.ipynb` created as the primary Kaggle workflow.
- The combined notebook contains 79 cells, including 49 code cells. Gradient boosting is the only supervised classifier. All code compiles, and all 45 non-export analytical cells pass in notebook order.
- The supervised split has 24,000 training rows, 3,000 validation rows, and 3,000 test rows. Target prevalence is approximately 22.12% in every partition.
- Pairwise identifier overlap is zero. Maximum absolute standardized mean difference across all feature and partition comparisons is 0.0742, below the common 0.10 imbalance warning threshold.
- All sex, education, marriage, and age-band groups occur in every partition. Some extremely rare repayment-status codes are absent from validation or test. Education-code-5/default-class-1 is absent from test, and marriage-code-0/default-class-1 is absent from validation and test. The notebook reports these gaps explicitly.
- Ten test rows share a 23-feature profile with at least one development row, but identifiers remain distinct. Twenty-one feature profiles have conflicting target labels, demonstrating that the available features do not determine the target perfectly.
- Under the current split, K-Means selects 2 clusters in the 10-component PCA representation with mean initialization-stability ARI 0.996. Clusters remain behavioral segments rather than risk groups.
- No tested DBSCAN candidate produces multiple substantial density clusters. The current anomaly-exploration configuration uses 5 whitened PCA components, `eps` 1.1526, and `min_samples` 10. It identifies one core group and approximately 2.08% density-noise records.
- DBSCAN noise records represent unusual multivariate repayment and billing patterns only. They are not evidence of fraud, wrongdoing, or future default.
- The executed project notebook contains the canonical 80/10/10 Kaggle run with 49 executed code cells, 19 PNG outputs, and zero saved error outputs.
- The selected classifier uses learning rate 0.05, maximum 600 boosting iterations, 15 leaf nodes, minimum leaf size 20, and L2 regularization 1.0.
- Early stopping uses a 10% internal training-fold validation fraction, patience 30, tolerance `1e-7`, and log loss. The best internal validation iterations were 81, 72, and 138. The final calibrated estimators stopped at 116, 131, and 106 iterations.
- The notebook plots training and internal-validation log loss for each calibration fold and exports the curves and early-stopping summary as CSV files.
- The validation-selected threshold is 0.2916. On the final test set it produces accuracy 0.8047, balanced accuracy 0.7102, precision 0.5609, recall 0.5407, F1 0.5506, ROC-AUC 0.7877, PR-AUC 0.5713, and Brier score 0.1323.
- At the default 0.50 threshold, accuracy is 0.8247, precision is 0.6983, recall is 0.3660, and F1 is 0.4802. Both operating points remain visible in the notebook.
- The selected-threshold confusion matrix contains 2,055 true negatives, 281 false positives, 305 false negatives, and 359 true positives.
- Reusable dataset normalization and validation module created.
- Dataset validation unit tests and dependency file created.
- Live UCI retrieval verified locally: 30,000 rows, 23 features, no missing values, no duplicate identifiers, 23,364 class 0 records, and 6,636 class 1 records.
- The public portfolio package includes an executed notebook, 13 exported result figures, a generated README banner, unit tests, and a presentation source for NotebookLM or Gemini.
- Raw dataset files, trained model binaries, temporary caches, frontend placeholders, API placeholders, and deployment configuration are excluded from the repository.

## Next recommended task

Maintain the GitHub repository as the canonical portfolio artifact. When the notebook changes, rerun it in Kaggle, confirm the split and evaluation audits, export the updated figures with `ml/scripts/export_notebook_figures.py`, synchronize the README metrics, and regenerate the presentation from `docs/presentation-source.md`.
