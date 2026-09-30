# CS-334 Machine Learning — Lecture 10: Model Selection
**Date:** 09/29/2026 · **Instructor:** Prof. Shengpu Tang
**Slides:** `lectures/slides/Lecture-10-Model-Selection.pdf` (includes In-Class Exercise #8)
**Prereq:** `Lecture-09-Model-Assessment.md`

---

## Admin / Logistics

- **HW2** is in. Solutions posted Wednesday; grading hopefully by Thursday.
- **HW3 released** — "Predicting Survival of ICU Patients", issued Fri 9/25, **due Sun 10/11, 11:59pm.** Two Gradescope parts: **HW3-Written** (PDF; the spec highlights the sub-questions that must appear in the write-up) and **HW3-Code&Challenge** (`hw3_main.py`, `hw3_challenge.py`, `challenge.csv` with predictions for the held-out patients). The challenge leaderboard is disabled for the duration of the homework.
- **HW3's honor-code statement is stricter than HW2's:** *"THIS HOMEWORK IS MY OWN WORK, WRITTEN WITHOUT COPYING FROM OTHER STUDENTS **OR DIRECTLY FROM LARGE LANGUAGE MODELS SUCH AS CHATGPT**."* — plus an acknowledgment of any collaboration or external resources.
- **HW2 + HW3 extra credit** due **10/19** (replaces the 10/14 date given in Lecture 09).

---

## 0. Where This Lecture Sits

Lecture 09 answered **"what number do I compute?"** (the metrics). This lecture answers the other two questions:

1. **Model assessment, process:** *which data* do I compute it on? (holdout, K-fold CV, Monte Carlo CV, bootstrap)
2. **Model selection:** how do I pick hyperparameters like $\lambda$ — and how do I do that **without contaminating** the assessment?

| | Metrics | Process |
|---|---|---|
| Classification | accuracy, precision, recall, sensitivity, specificity, TPR, FPR, F1, AUROC, AUPRC, calibration* | training/test split (holdout) · K-fold CV, Monte Carlo CV · bootstrap CIs* |
| Regression | MSE, RMSE, MAE, MBE | |

`*` = not on the exam.

---

## 1. Metrics Recap (slides 5–13)

The Lecture 09 table, now with **FNR** added:

| Formula | Name |
|---|---|
| $(TP+TN)/N$ | accuracy |
| $TP/(TP+FN)$ | TPR, sensitivity, recall |
| $TN/(TN+FP)$ | TNR, specificity |
| $FP/(TN+FP)$ | FPR, 1 − specificity |
| $FN/(TP+FN)$ | **FNR, 1 − sensitivity** |
| $TP/(TP+FP)$ | precision, PPV |

The Venn-style picture on slide 5 is a good way to remember it: actual positives/negatives are one split of the data, predicted positives/negatives are a second, tilted split; the four overlaps are TP, FN, FP, TN. **Can we optimize all at once? No** → composite metrics (balanced accuracy, F-scores) or threshold-free curves (AUROC — random baseline **0.5**; AUPRC — random baseline **% positive**). See Lecture 09 §4–§8, including the worked ROC example.

### 1.1 Metrics in sklearn

`sklearn.metrics`: `confusion_matrix`, `classification_report`, `precision_score`, `recall_score`, `f1_score` take **hard labels**; `precision_recall_curve`, `roc_curve`, `roc_auc_score` take **scores**.

```python
y_pred  = clf.predict(X)                  # predicted class {−1,+1}    — ONE threshold
y_prob  = clf.predict_proba(X)[:, 1]      # σ(θ⃗·x⃗), P(y = +1)          — ALL thresholds
y_score = clf.decision_function(X)        # θ⃗·x⃗, continuous output    — ALL thresholds
```

> **The classic bug:** passing `y_pred` to `roc_auc_score`. With hard labels there's only one threshold, so the "curve" is a single point joined to the corners and the AUROC is just balanced accuracy. Curves and AUCs need `predict_proba(X)[:, 1]` (column 1 = the positive class) or `decision_function(X)`. Either works for AUROC, since it only depends on the ranking and $\sigma$ is monotone.

### 1.2 New: $R^2$, the coefficient of determination

$$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$

- **"Goodness of fit"** measure.
- **Interpretation: the proportion of variability in $y$ explained by the model.** The denominator is the error of the dumbest possible regressor — always predict the mean $\bar y$ — so $R^2$ says how much better than that you are.
- Scatterplots on the slide: $r^2 = 0$ (a cloud), $0.5$ (a noisy trend), $1.0$ (points exactly on a line).
- **Slide:** "always lies between 0 and 1." That holds for OLS **with an intercept, on its own training data**. On **test** data $R^2$ can be **negative** — any model worse than predicting $\bar y$ — and sklearn's `r2_score` / `.score()` will happily return negative values. Upper bound 1 always holds (perfect fit).

---

## 2. Why You Can't Assess on the Training Data

> **Use all the data to train the model and report the performance on all the data?** — No.

**The pitfall of training error:** when the model is overly complex, training error can be close to **0%**:

- a **lookup table** (memorize every $(\vec x, y)$ pair),
- **kernelized classifiers with no regularization** (Lec 11),
- a **high-degree polynomial** (Lecture 07's $M=9$ fit).

The model **"memorizes" the training data but does not generalize to new data.** The handwritten plot: as model complexity grows, **training error** falls toward 0 while **test error (generalization error)** is U-shaped — the red circle marks the far right, where the gap is largest. Training error is a biased, optimistic estimate of what we actually care about.

---

## 3. Holdout: A Test Set

- **Hold out some data (the test data) that is not used for training** the model.
- The test set is a **proxy for "everything you might see."**

```
Training data ─ X_train, Y_train ─►  SVC().fit(X_train, Y_train)  ─► model
Test data     ─ X_test ──────────►   model.predict(X_test)        ─► Y_predict
              └ Y_test ─────────►    error_metric(Y_test, Y_predict) ─► TEST ERROR
```

### 3.1 How big should the test set be?

- **Too few for training** → unable to properly learn from the data.
- **Too few for testing** → bad approximation of the true error (a noisy estimate).
- **Rule of thumb:** enough test samples to form a reasonable estimate.
- **Common splits: 70%–30% or 80%–20%.**

**What to do when there isn't enough data?** A single holdout "wastes" the test data — it's never trained on — and with small $N$ the one estimate depends heavily on which points happened to land in the test set. → cross-validation.

---

## 4. Cross-Validation

### 4.1 K-fold CV

**Use all the data to train/test — but don't use all the data to train at a time.**

1. Split the data into $K$ parts, or **"folds."**
2. Train on all but the $k$-th part; test/validate on the $k$-th part.
3. Repeat for each $k = 1,\dots,K$.
4. **Report the average over the $K$ experiments.**

Every example is in the test fold **exactly once**, and in the training set $K-1$ times.

### 4.2 Common values of $K$

| $K$ | Name | Notes |
|---|---|---|
| 2 | two-fold CV | each model trains on only half the data |
| **5, 10** | 5-fold, 10-fold | **common practice** |
| $N$ | **leave-one-out (LOOCV)** | test on one example at a time; $N$ models to train |

**Selection is based on how much data you have.** The trade-off: bigger $K$ → each training set is closer to the full dataset (less pessimistic estimate), but you train more models, and the $K$ models become near-identical.

### 4.3 Monte Carlo CV

**A.k.a. repeated random sub-sampling.**

- **Randomly select (without replacement)** some fraction of the data to form the training set; assign the rest to the test set.
- **Repeat multiple times** with different partitions.

The difference from K-fold: the test sets are **not disjoint** — an example can be tested several times or never. You choose the number of repetitions and the split fraction independently.

### 4.4 Group activity — comparing the three

| | **Holdout** | **K-fold CV** | **Monte Carlo CV** |
|---|---|---|---|
| How many possible partitions do we see? | **1** | **$K$** | **as many as possible** |
| How many models do we need to train? | **1** | **$K$** | **one per repetition** (repeat 1000× → 1000 models) |
| What's the "final model"? | **the model** | **average (?)** | **average (?)** |
| How to get error bars? | **bootstrap\* the test set** | **over the $K$ folds** | **over the 1000 repetitions** |

> **About the "(?)"** — you don't literally average $K$ sets of weights. The CV score estimates how well **the procedure** ("train *this* kind of model with *these* hyperparameters on this much data") generalizes. The usual final step is to **retrain once on all the data** with that procedure and ship that model; the CV average is your estimate of its performance.

---

## 5. *Bootstrapped Confidence Intervals (not on the exam)

- We want to know **how much performance is likely to vary** — but in many settings we have only **a single test set**.
- **Bootstrap (Efron 1979):** given a test set of size $N$, **sample $N$ times with replacement** to create a new test set, compute the metric on it, and repeat (e.g. 1000 times).
- Take the **2.5th and 97.5th percentiles** of the 1000 values → **95% CI [lower, upper]**.
- The statistic doesn't have to be an average — it can be **any** sample statistic: AUROC, accuracy, F1, …

No retraining is involved: the model is fixed; only the evaluation set is resampled. That's why it's the "error bars" answer for **holdout** in the §4.4 table.

---

## 6. Assessment: The Process

```
Data ─► feature extraction & preprocessing ─► ┌ Training ┐─ h(x⃗) ─► Model ─► ŷ ─┐
                                              └ Test ────┘─────────────▲        ▼
                                                   └──── y ─────────►  QUALITY METRIC
```

Split with **holdout, K-fold CV, or Monte Carlo CV**, and **report the performance on the "test" data.**

---

## 7. Model Selection: Hyperparameters

**Model selection = selecting the proper level of flexibility for a model** — a **"meta-optimization"** over **hyperparameters**:

| Hyperparameters (chosen *by us*, before training) | Parameters (learned *by training*) |
|---|---|
| regularization strength ($\lambda$, or `C`) | feature coefficients $\vec\theta$ (and $b$) |
| regularization type (L1 or L2) | |
| loss function (logistic regression vs. SVM) | |
| polynomial degree $M$ | |
| kernel type (linear vs. RBF) | |
| (later: $K$ for KNN, tree size for decision trees) | |

The distinction: `.fit()` **optimizes the parameters for fixed hyperparameters**. You can't fit hyperparameters on the training set — e.g. $\lambda = 0$ always minimizes training error — so they need **held-out data**. HW2 Q3 did this by hand: choose $M$ and $\lambda$ by RMSE on a separate validation set.

### 7.1 K-fold CV for hyperparameter selection — "simple, popular solution"

For each candidate $C_1,\dots,C_5$, run full K-fold CV:

```
             C₁                C₂           …      C₅
fold 1   Model(C₁,train1)  Model(C₂,train1)
fold 2   Model(C₁,train2)  Model(C₂,train2)
  ⋮
fold 5   Model(C₁,train5)  Model(C₂,train5)
         ─────────────     ─────────────
         AvgPerf(C₁)       AvgPerf(C₂)     …   AvgPerf(C₅)
                                 └──── Best C = argmax ────┘
```

**Models trained: (number of candidates) × $K$.**

---

## 8. Doing Selection *and* Assessment

**One idea:** tune the hyperparameters **and** report the performance using the same K-fold CV. **Is this a good idea? No.**

The reported number is the **maximum** of several noisy CV scores — you picked the $C$ whose folds happened to look best — so it's **optimistically biased**. The hyperparameter has, in effect, been fitted to the test folds. The more candidates you try, the worse the bias.

### Guidelines

- **Don't use the same samples to choose optimal hyperparameters and to estimate test/generalization error.**
- The choice of methodology depends on your problem and dataset.

### 8.1 Three-way split

```
Available data ─► [        Training        ][ Testing (holdout sample) ]
                   └► [ Training ][ Validation (validation holdout) ]
```

- **Validation data will NOT be used for training the model.**
- **Test data will NOT be used for training the model OR tuning hyperparameters.**

Process (six steps, from Raschka's blog):
1. Split data into training / validation / test.
2. Train one model per hyperparameter setting on the **training** set.
3. Evaluate each on the **validation** set; pick the best.
4. **Can merge training + validation** and retrain with the best hyperparameters.
5. Evaluate that model **once** on the **test** set → reported performance.
6. **Can use all the data** (train + val + test) to train the **final** model you deploy.

### 8.2 Holdout + K-fold CV

- **Holdout:** a test set to **assess** the performance.
- **Training set:** use **K-fold CV** to **find the optimal hyperparameter**:
  $$C^* = \arg\min_{C}\ \widehat{\text{Err}}_C, \qquad \widehat{\text{Err}}_C = \text{average error over the } K \text{ folds}$$
- Use the optimal hyperparameters to **train on the full training data and assess on the test set.**

Same six steps as §8.1, with step 2–3 replaced by K-fold CV on the training set. This is the pipeline **HW3** asks for, and the one In-Class Exercise #8 describes.

### 8.3 Nested CV

Replace the single holdout with an **outer** CV loop too:

- **Outer $k$-fold loop:** **assess** the performance — $T$ partitions $(\text{Training}_j, \text{Test}_j)$.
- **Inner $k$-fold loop:** inside each $\text{Training}_j$, **choose the optimal hyperparameter** $C_1,\dots,C_M$ by K-fold CV.

Each outer fold may pick a *different* $C^*$ — that's fine; nested CV assesses **the whole procedure, tuning included**. It gives the least-biased estimate and uses all data for testing, at the cost of many more models: roughly $T \times (M \times K + 1)$.

| Setup | Unbiased assessment? | Models trained ($M$ candidates, $K$ folds) | Use when |
|---|---|---|---|
| CV for both (§8 "one idea") | **no** — optimistic | $MK$ | never, for reporting |
| Three-way split | yes | $M + 1$ (+1 final) | lots of data |
| Holdout + K-fold CV | yes | $MK + 1$ | the usual default |
| Nested CV | yes, least variance | $T(MK + 1)$ | small data |

---

## 9. sklearn

Most commonly used: `train_test_split`, `KFold`, `StratifiedKFold` (in `sklearn.model_selection`).

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9)
```

→ **80-10-10 train-val-test three-way split.** The second call takes $1/9$ of the remaining 90%, which is 10% of the original.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

→ **5-fold split in a for-loop.** **Stratify = maintain label proportions** in every fold. Important for imbalanced data like HW3's: with plain `KFold`, a fold could end up with almost no positives, and AUPRC/AUROC on that fold would be meaningless. Note `skf.split` needs `y` (to stratify) and returns **indices**, not arrays.

### 9.1 Hyperparameter search

- **`GridSearchCV`** — **exhaustive** search over every combination, scored by CV.
- **`RandomizedSearchCV`** — **randomized** search: a fixed number of samples (`n_iter`) from the parameter space.

```python
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(
    LogisticRegression,               # (as on the slide — in real code, an instance: LogisticRegression())
    {'C': [0.1, 1, 10, 100]},
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5),
)
```

- **HW3 Q2 — implement your own grid search with cross-validation.** The skeleton has two functions: `cv_performance(clf, X, y, k=5, metric='accuracy')` (average a metric over stratified folds) and `select_C(X, y, C_range, penalty, k, metric)` (loop over `C_range`, call `cv_performance`, return the best `C`).
- **HW3 Q3 — feel free to use sklearn's grid/random search**; read the documentation.

Grid search with $d$ hyperparameters of $m$ values each costs $m^d$ settings; random search's cost is whatever `n_iter` you choose, independent of $d$.

---

## 10. In-Class Exercise #8 — "How many models are you training (for HW3)?"

**Q2 — The ML pipeline.** *"Logistic regression with elastic-net regularization. First split into training and test. 5-fold CV on the training set, grid search over $\lambda_1, \lambda_2 \in \{10^{-4}, 10^{-3}, \dots, 10^{5}\}$. The best setting (highest average CV AUROC) is used to train a final model on the entire training set, whose performance is reported on the test set."*

- Each $\lambda$ has **10** values ($10^{-4}$ through $10^{5}$ — count the exponents: $-4,\dots,5$).
- Grid: $10 \times 10 = 100$ settings.
- CV: $100 \times 5 = 500$ models.
- Plus **1** final model on the full training set.
- **Total: 501.**

**Q3 — Random search with 10 trials instead.** $10 \times 5 = 50$ CV models $+ 1$ final $=$ **51**.

**Q4 — "Nested CV" instead of "Holdout + CV": fewer or more models?** **More.** The whole 501-model procedure runs once per outer fold: with 5 outer folds, $5 \times 501 = 2505$ (plus one more tuning run on all the data if you then want a single deployed model).

> **The pattern:** count settings × folds, then **don't forget the +1 refit** after selection. It's the step people drop.

---

## Quick Reference

```
ASSESSMENT = evaluate performance        SELECTION = choose flexibility (hyperparameters)
  (* = not on exam: calibration, bootstrap CIs)

METRICS RECAP   FNR = FN/(TP+FN) = 1 − sensitivity
  sklearn:  predict(X)            → labels, ONE threshold   (confusion_matrix, f1_score, …)
            predict_proba(X)[:,1] → σ(θ⃗·x⃗)  ┐ ALL thresholds
            decision_function(X)  → θ⃗·x⃗     ┘ (roc_curve, roc_auc_score, PR curve)
  R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ)²  = fraction of variance in y explained
       ≤ 1; in [0,1] for OLS on its training data; CAN BE NEGATIVE on test data

TRAINING ERROR  → ~0 for lookup tables, unregularized kernels, high-degree polys
                  model memorizes, doesn't generalize → never report it

HOLDOUT   train/test, typically 70/30 or 80/20. 1 partition, 1 model, error bars via bootstrap*
K-FOLD    K folds, each tests once; K models; report avg (± over folds)
          K = 5, 10 common;  K = N is LOOCV.  choose K by how much data you have
MONTE CARLO CV  repeated random subsampling (w/o replacement); test sets overlap
                one model per repetition; error bars over repetitions
"final model"   retrain on all data with the chosen procedure; CV avg estimates its perf.
BOOTSTRAP*  resample TEST set N times WITH replacement, ×1000; 2.5 / 97.5 pctl = 95% CI

HYPERPARAMETERS (λ/C, L1 vs L2, loss, poly degree, kernel) ≠ PARAMETERS (θ⃗, b)
K-FOLD SELECTION   for each candidate: K-fold CV → AvgPerf;  best = argmax.  M·K models

DON'T use the same samples to pick hyperparameters AND estimate generalization error
  (max of noisy CV scores is optimistically biased)
THREE-WAY SPLIT   train / validation / test.  val: no training.  test: no training, no tuning
HOLDOUT + CV      test held out; K-fold CV on train → C* = argmin Err_C;
                  refit on all train; evaluate ONCE on test.       M·K + 1 models
NESTED CV         outer loop assesses, inner loop selects.         ≈ T·(M·K + 1) models
after assessment: may retrain on ALL data for the deployed model

SKLEARN   train_test_split(X, y, test_size=0.1) then (…, test_size=1/9) → 80/10/10
          StratifiedKFold(n_splits=5).split(X, y) → index pairs; keeps label proportions
          GridSearchCV (exhaustive, m^d settings) · RandomizedSearchCV (n_iter samples)
          HW3 Q2: write your own CV grid search.  HW3 Q3: sklearn's is fine.

EXERCISE #8   10 × 10 grid × 5 folds + 1 refit = 501 · random 10 trials: 10×5 + 1 = 51
              nested CV → MORE models
```
