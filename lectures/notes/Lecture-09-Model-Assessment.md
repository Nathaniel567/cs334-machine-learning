# CS-334 Machine Learning — Lecture 09: Model Assessment
**Date:** 09/24/2026 · **Instructor:** Prof. Shengpu Tang
**Slides:** `lectures/slides/Lecture-09-Model-Assessment.pdf`
**Prereq:** `Lecture-08-Logistic-Regression.md`

---

## Admin / Logistics

- **HW2** due **Sunday 9/27, 11:59pm.** Solutions released the following Wednesday; grades Thursday or the week after.
- **HW3 — "Predicting Survival of ICU Patients"** released Fri 9/25, due in two weeks (**Sun 10/11, 11:59pm**). Topics: Feature Engineering (Lec 2, 11), Logistic Regression (Lec 8), Model Assessment (Lec 9), Model Selection (Lec 10), Feature Selection (Lec 11).
- **HW3 extra credit: the ROC curve (this lecture).** Originally announced as due Wed 10/14; Lecture 10 moved it to **10/19** (combined HW2 + HW3 extra credit).

### HW3 at a glance

- **Data:** PhysioNet ICU records — time-stamped measurements from the first **48 hours** of each admission (age, height, heart rate, temperature, respiratory rate, …). Each patient's time series gets summarized into a fixed-length feature vector like $\vec x^{(i)} = [56.0,\ \text{NaN},\ 84.245,\ 37.093,\ \text{NaN},\ \dots]^T$ — **note the NaNs**; missing values are part of the problem.
- **Output:** $\hat y = P(\text{death})$, e.g. 20% — a *probability*, so this is logistic regression from Lecture 08, evaluated with the metrics from this lecture.
- **Grading:**

| Points | Part |
|---|---|
| **80** | Guided process of building a full ML pipeline — **18** autograded code, **62** written answers |
| **20** | **Open-ended challenge** on a set of **unseen** patients (features only): **10** written (describe what you did), **10** **graded on prediction quality** |

---

## 0. Where This Lecture Sits

The handwritten course map (slides 5–6) is worth copying down once, because every lecture so far slots into it:

```
PROBLEM SETUP           minimize regularized empirical risk
  classification ─┐       loss function          regularization
  regression     ─┘       ▸ zero-one loss         ▸ L2
                          ▸ perceptron loss       ▸ L1
                          ▸ logistic loss         ▸ ElasticNet
                          ▸ hinge loss
                          ▸ squared loss          Trade-off: bias vs. variance

OPTIMIZATION TOOLS      LINEAR MODELS
  ▸ SGD                   perceptron · logistic regression · SVM* · ridge · LASSO
  ▸ Newton's method*
  ▸ closed-form solution
  ▸ Lagrangian dual*

                SUPERVISED LEARNING: x → y     Goal: GENERALIZATION to unseen data
```

The circled word is **generalization**. Everything up to Lecture 08 was about *fitting* a model; this lecture and the next are the first two items under **Practical Aspects**: **model assessment**, **model selection** (then feature selection, Lec 11). Greyed out on the map, still to come: non-linear models (explicit feature maps $\phi(\vec x)$, kernels, decision trees, random forests/bagging, boosting, neural networks), and unsupervised / reinforcement learning.

Two definitions frame both lectures:

- **Model assessment** — evaluating a model's performance.
- **Model selection** — selecting the proper level of flexibility for a model (e.g. regularization strength in logistic regression). → Lecture 10.

> *"Remember that all models are wrong; the practical question is how wrong do they have to be to not be useful."* — George Box, 1987

---

## 1. sklearn Recap and One Practical Trap

Slides 7–8 repeat the Lecture 08 tables (`LinearRegression`, `Ridge`, `Lasso` = `ElasticNet` without L2, `SGDRegressor`; `LogisticRegression`, `SGDClassifier`, `Perceptron`, `LinearSVC`/`SVC`). New here is how the pieces line up with the math:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)            # θ⃗*, b* = argmin_{θ⃗,b}  R_N(θ⃗) + λΩ(θ⃗)
y_pred = clf.predict(X_test)         # h(x⃗; θ⃗, b) = σ(θ⃗·x⃗ + b)  → thresholded to a label
np.sum(y_test == y_pred) / N_test    # accuracy
```

### 1.1 sklearn's objective is scaled differently

| | Objective |
|---|---|
| What we saw in class | $\displaystyle \frac{1}{N}\sum_{i=1}^{N}\text{loss}_{\log}\big(y^{(i)},\hat y^{(i)}\big) + \lambda\,\Omega(\vec\theta)$ |
| What sklearn uses | $\displaystyle C\sum_{i=1}^{N}\text{loss}_{\log}\big(y^{(i)},\hat y^{(i)}\big) + \Omega(\vec\theta)$ |

Divide the class objective by $\lambda$ (which doesn't move the minimizer) and you get sklearn's form with

$$C = \frac{1}{N\lambda}.$$

So "$C = 1/\lambda$" is right **up to the $1/N$**: sklearn *sums* the loss instead of averaging it. Practical consequence — the same `C` regularizes a big dataset *less* than a small one, because the data term grows with $N$ while $\Omega$ doesn't.

### 1.2 Not every solver supports every penalty

| `solver` | supported `penalty` |
|---|---|
| `'lbfgs'` (default) | `'l2'`, `None` |
| `'liblinear'` | `'l1'`, `'l2'` |
| `'newton-cg'` | `'l2'`, `None` |
| `'newton-cholesky'` | `'l2'`, `None` |
| `'sag'` | `'l2'`, `None` |
| `'saga'` | `'elasticnet'`, `'l1'`, `'l2'`, `None` |

For L1 you need `liblinear` or `saga`; for elastic net, **only `saga`**. The defaults (`penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `max_iter=100`) mean an out-of-the-box `LogisticRegression()` is already L2-regularized.

---

## 2. The Evaluation Pipeline

```
Data (x⃗ᵢ, yᵢ) ──► feature extraction & preprocessing ──► h(x⃗) ──► Model ──► ŷ ─┐
      │                                                                       ▼
      └────────────────────────── y ───────────────────────────────►  QUALITY METRIC
```

A quality metric compares predictions $\hat y$ against true labels $y$. The lecture splits the topic into **what** to measure and **how** to get data to measure it on:

| | Classification | Regression |
|---|---|---|
| **Metrics** | accuracy, precision, recall, sensitivity, specificity, TPR, FPR, F1, AUROC, AUPRC, calibration* | MSE, RMSE, MAE*, MBE* |

| **Process** |
|---|
| training/test split (holdout) · K-fold CV, Monte Carlo CV · bootstrap confidence intervals* |

`*` = not responsible for on the exam. This lecture covers the metrics; the process is Lecture 10. (Lecture 10 shows this slide twice — once without the asterisks on MAE and MBE, once with them. Learn them anyway; they're one line each.)

---

## 3. Accuracy — and What's Wrong With It

Given $\hat y = h(\vec x;\vec\theta)$ and $\mathcal D = \{(\vec x^{(i)}, y^{(i)})\}_{i=1}^N$, $y^{(i)}\in\{-1,+1\}$:

$$\text{Misclassification error} = \frac1N\sum_{i=1}^N \mathbb 1\big[\hat y^{(i)}\ne y^{(i)}\big] \qquad \text{Accuracy} = \frac1N\sum_{i=1}^N \mathbb 1\big[\hat y^{(i)} = y^{(i)}\big] = 1 - \text{error}$$

**What's wrong with accuracy?**

- **Inflated under class imbalance.** Rare disease with 1% prevalence: the model "always predict healthy" scores **99%** accuracy while catching zero sick patients. The **base-case accuracy** (always predict the majority class) can be very high, so a high number on its own means nothing — always compare to it.
- **Assumes equal cost for both kinds of error.** Missing fraud and flagging a legitimate purchase are counted the same, which is rarely true.

The fraud example (slides 16–19) motivates splitting "wrong" into two kinds: predicted *not fraud* but actually fraud is a **false negative**; predicted *fraud* but actually not is a **false positive**. Likewise "right" splits into **true negatives** and **true positives**.

---

## 4. The Confusion Matrix

|  | **Predicted (−)** | **Predicted (+)** |
|---|---|---|
| **Actual (−)** | True Negative (TN) | False Positive (FP) |
| **Actual (+)** | False Negative (FN) | True Positive (TP) |

How to read the names: the **second word is what the model predicted**, the **first word is whether it was right**. A *false positive* predicted positive and was wrong.

The slides' memorable version (pregnancy test): a **false positive** tells a man "you're pregnant" — **Type I error**; a **false negative** tells a visibly pregnant woman "you're not pregnant" — **Type II error**.

### 4.1 The metrics

$P = TP + FN$ (actual positives), $N_- = TN + FP$ (actual negatives), $N = P + N_-$.

| Formula | Names | Denominator = |
|---|---|---|
| $(TP+TN)/N$ | **accuracy** | everything |
| $TP/(TP+FN)$ | **true positive rate (TPR)**, **sensitivity**, **recall** | actual positives |
| $TN/(TN+FP)$ | **true negative rate (TNR)**, **specificity** | actual negatives |
| $FP/(TN+FP)$ | **false positive rate (FPR)**, $1-$specificity | actual negatives |
| $FN/(TP+FN)$ | **false negative rate (FNR)**, $1-$sensitivity | actual positives |
| $TP/(TP+FP)$ | **precision**, **positive predictive value (PPV)** | **predicted** positives |

> **The denominator is the whole trick.** TPR, FNR divide by the actual-positive *row*; TNR, FPR divide by the actual-negative *row*; precision is the odd one out and divides by the predicted-positive *column*. Recall asks *"of the sick, how many did I catch?"*; precision asks *"of the ones I flagged, how many were sick?"*

### 4.2 The three questions on the slide

**Range?** Every one of these is a fraction of counts, so each lies in $[0,1]$.

**Predict everything positive** (no FN, no TN):
TPR = recall = **1**, specificity = **0**, FPR = **1**, precision = $P/N$ = the **positive rate** of the data, accuracy = $P/N$.

**Predict everything negative** (no TP, no FP):
TPR = **0**, specificity = **1**, FPR = **0**, precision = $0/0$ — **undefined** (sklearn returns 0 with a warning), accuracy = $N_-/N$.

**Can we optimize all metrics at once?** **No.** The two trivial classifiers each get a perfect score on one side and a zero on the other. Sensitivity trades against specificity, recall against precision — which is why we have composite metrics and threshold curves.

### 4.3 Composite metrics

**Balanced accuracy** — mean of sensitivity and specificity:

$$\text{BA} = \frac12\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$$

Both trivial classifiers score exactly **0.5** — fixing the rare-disease problem above.

**F1-score** — **harmonic** mean of precision and recall:

$$F_1 = \frac{2}{\frac{1}{TPR}+\frac{1}{PPV}} = 2\,\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
\qquad
F_\beta = (1+\beta^2)\,\frac{\text{Precision}\cdot\text{Recall}}{\beta^2\,\text{Precision}+\text{Recall}},\ \ \beta>0$$

Why *harmonic*: it's dominated by the smaller of the two. Precision 1.0 with recall 0.01 has arithmetic mean ≈ 0.5 but $F_1 \approx 0.02$ — you can't buy a good F1 by maxing out one side. $F_\beta$ weights recall $\beta$ times as much as precision: $\beta>1$ favors recall (screening), $\beta<1$ favors precision.

Note F1 **never uses TN** — it only cares about the positive class, which is what you want when negatives are plentiful and uninteresting.

---

## 5. Discrimination Thresholds

Two ways to output a prediction from the same $\vec\theta$:

| | Output | Rule |
|---|---|---|
| Predicted class | $\{-1,+1\}$ | $h(\vec x;\vec\theta) = \mathrm{sign}(\vec\theta\cdot\vec x)$ |
| Predicted class probability | $[0,1]$ | $h(\vec x;\vec\theta) = \sigma(\vec\theta\cdot\vec x)$ |

Normally we output **+1 if probability > 0.5**, i.e. $\vec\theta\cdot\vec x > 0$ — the solid line in the slide's picture. **What if we use a different threshold?** Requiring $\sigma(\vec\theta\cdot\vec x) \ge 0.8$ gives the dashed line: **parallel** to the original (same $\vec\theta$, just a shifted offset), and further into the $\times$ region. Fewer points get called positive — fewer false positives, more false negatives.

Every threshold is a different classifier with its own confusion matrix. Rather than pick one, sweep them all.

---

## 6. The ROC Curve

**Receiver Operating Characteristic** curve:

- **y-axis: true positive rate** (sensitivity)
- **x-axis: false positive rate** (1 − specificity)
- **Each point = one threshold / one decision boundary**, i.e. one trade-off between FPR and TPR.

### 6.1 Building one by hand (slides 26–35)

Sort by predicted probability, then move the threshold down one example at a time. Everything **above** the line is predicted positive.

| $\hat y$ | 90% | 80% | 70% | 60% | 40% | 30% | 20% | 10% |
|---|---|---|---|---|---|---|---|---|
| $y$ | +1 | +1 | −1 | +1 | −1 | −1 | +1 | −1 |

4 positives, 4 negatives, so each TP moves TPR up by **0.25** and each FP moves FPR right by **0.25**.

| Threshold between… | predicted + | TP | FP | **FPR** | **TPR** |
|---|---|---|---|---|---|
| above 90% | none | 0 | 0 | 0 | 0 |
| 90 / 80 | 1 | 1 | 0 | 0 | 0.25 |
| 80 / 70 | 2 | 2 | 0 | 0 | 0.5 |
| 70 / 60 | 3 | 2 | 1 | 0.25 | 0.5 |
| 60 / 40 | 4 | 3 | 1 | 0.25 | 0.75 ← the usual 0.5 threshold |
| 40 / 30 | 5 | 3 | 2 | 0.5 | 0.75 |
| 30 / 20 | 6 | 3 | 3 | 0.75 | 0.75 |
| 20 / 10 | 7 | 4 | 3 | 0.75 | 1 |
| below 10% | all 8 | 4 | 4 | 1 | 1 |

> **The shortcut:** walk down the sorted list — a **positive** steps **up** by $1/P$, a **negative** steps **right** by $1/N_-$. The ROC curve is a staircase from $(0,0)$ (threshold above everything — predict nothing positive) to $(1,1)$ (predict everything positive). Ties in score produce a diagonal step. (The slides build it in the reverse direction, starting from $(1,1)$; same curve.)

At the default 0.5 threshold (between 60% and 40%): TP 3, FP 1, FN 1, TN 3 → accuracy 0.75, TPR 0.75, FPR 0.25, precision 0.75.

**This is the HW3 extra credit.** Checked with `sklearn.metrics.roc_curve` (`drop_intermediate=False`) — same nine points.

### 6.2 Reading ROC curves

- **Always increasing** (slope ≥ 0) — lowering the threshold can only add predicted positives, so TP and FP can only go up.
- **A: perfect predictions** — straight up the left edge to $(0,1)$, then across.
- **B: random predictions** — the diagonal $TPR = FPR$.
- **Two non-intersecting curves** → the higher one **dominates** (better at every FPR).
- Shows the sensitivity/specificity trade-off, **but isn't a good summary metric — it's not a single number.**

---

## 7. AUROC

**Area Under the ROC Curve** (AUROC, ROC-AUC), computed with the **trapezoid rule** — `sklearn.metrics.auc(x, y)` (or directly `roc_auc_score(y_true, y_score)`).

**Intuitive meaning:** pick one random positive and one random negative example; AUROC is the probability the model scores the positive higher:

$$\text{AUROC} = P\big[\text{score}(x^+) > \text{score}(x^-)\big]$$

Worked example from §6.1: area $= 0.25(0.5) + 0.25(0.75) + 0.25(0.75) + 0.25(1) = \mathbf{0.75}$. Pairwise check: positives at 90, 80, 60, 20 beat 4, 4, 3, 1 of the negatives (70, 40, 30, 10) → $12/16 = 0.75$. ✓

> **Why the two views agree:** each positive is a vertical step, each negative a horizontal one. The area to the left of a negative's horizontal step is exactly the fraction of positives ranked above it. Summing over negatives counts correctly ordered pairs.

| AUROC | Interpretation |
|---|---|
| **1** | perfect prediction |
| **> 0.9** | excellent — **something is potentially fishy; check for information leakage** |
| **0.8** | good |
| **0.5** | random |
| **< 0.5** | something is wrong! (flipping the predictions would score $1-\text{AUROC}$) |

Two things AUROC **ignores**: the threshold (it summarizes all of them), and the actual probability values — only the **ranking** matters. Any monotone transform of the scores (e.g. using $\vec\theta\cdot\vec x$ instead of $\sigma(\vec\theta\cdot\vec x)$) gives the same AUROC.

---

## 8. Precision–Recall Curve and AUPRC

- Built the same way, **sweeping the classification threshold**, but plotting **precision (y) vs. recall (x)**.
- **High AUPRC = both high recall and high precision.**
- **Random baseline: AUPRC = % positive** (a flat line at the positive rate) — not 0.5.
- Unlike ROC, the PR curve is **not monotone** — precision can go down *and* back up as the threshold moves.
- **In practice, report both.**

For the §6.1 example, sklearn's `average_precision_score` gives ≈ 0.83 (precision at each positive: 1, 1, 3/4, 4/7, averaged).

### 8.1 AUROC vs. AUPRC under imbalance

The slide's example: **20 positives, 2000 negatives.**

| | "Preferred" model | "Other" model |
|---|---|---|
| **ROC-AUC** | 0.87 | **0.91** |
| **AUPRC (AP)** | **0.77** | 0.28 |

ROC says the "other" model is slightly better; PR says it's far worse. Why: with 2000 negatives, 100 false positives is only FPR = 0.05 — a barely visible step on the ROC plot — but next to ≤ 20 true positives it tanks precision. **FPR is diluted by the huge TN count; precision never looks at TN.** When positives are rare and you care about them, AUPRC is the more honest number.

---

## 9. Selecting a Threshold

Deploying a model means picking **one** threshold. Pick it from a requirement, e.g. **"I want a minimum sensitivity of 80%"** → makes sure we catch 80% of the positive cases. Then read the cost off the curves:

- ROC: TPR = 0.8 at **FPR ≈ 5%** (ROC-AUC 0.930)
- PR: recall = 0.8 at **PPV ≈ 36%** (AUPRC 0.591)

Same threshold, two very different stories — only 5% of negatives get flagged, but **almost two-thirds of the flags are false alarms**. The ROC curve alone would hide that. (See sklearn's `TunedThresholdClassifierCV` / the "classification threshold" user-guide page.)

---

## 10. *Probability Calibration (not on the exam)

When the model outputs $\sigma(\vec\theta\cdot\vec x)$, is that number a meaningful probability? **If my model predicts 90% for a set of examples, do 90% of them have label +1?** "A prediction of 90% chance of spam should be spam 90% of the time."

**Calibration curve (reliability diagram):** bin examples by predicted probability; plot mean predicted probability (x) against the actual fraction of positives (y). **Well calibrated** hugs the diagonal; **poorly calibrated** is S-shaped or bowed.

Calibration and discrimination are separate: AUROC only looks at rankings, so a model can have excellent AUROC and terrible calibration. It matters for HW3, whose output is literally $P(\text{death})$.

---

## 11. Multiple Classes

With a $K\times K$ confusion matrix (the diagonal is correct), per-class $TP_k$, $FP_k$ (column $k$ minus diagonal), $FN_k$ (row $k$ minus diagonal):

$$\text{ACC} = \frac{TP_1+TP_2+TP_3}{\text{Total}}
\qquad
\text{PRE}_\text{macro} = \frac{\text{PRE}_1+\text{PRE}_2+\text{PRE}_3}{3}
\qquad
\text{PRE}_\text{micro} = \frac{TP_1+TP_2+TP_3}{TP_1+TP_2+TP_3+FP_1+FP_2+FP_3}$$

- **Macro** = compute the metric per class, then average — **every class counts equally**, so rare classes matter.
- **Micro** = pool the counts across classes, then compute — **every example counts equally**, so big classes dominate.
- **Micro-precision = micro-recall = accuracy.** Every misclassified example is exactly one class's FP (the one predicted) and exactly one class's FN (the true one), so $\sum FP_k = \sum FN_k = $ number of errors, and both denominators equal the total.

---

## 12. Regression Metrics

Given $\hat y = f(\vec x;\vec\theta)$, $y^{(i)}\in\mathbb R$:

| Metric | Formula | Range | Perfect | Notes |
|---|---|---|---|---|
| **MSE** | $\frac1N\sum(\hat y^{(i)}-y^{(i)})^2$ | $[0,\infty)$ | 0 | squared units; punishes big errors hard |
| **RMSE** | $\sqrt{\frac1N\sum(\hat y^{(i)}-y^{(i)})^2}$ | $[0,\infty)$ | 0 | **same units as $y$**; what HW2 plotted |
| **MAE** | $\frac1N\sum\lvert\hat y^{(i)}-y^{(i)}\rvert$ | $[0,\infty)$ | 0 | robust to outliers; $\text{MAE}\le\text{RMSE}$ always |
| **MBE** | $\frac1N\sum(\hat y^{(i)}-y^{(i)})$ | $(-\infty,\infty)$ | 0 | **sign** = systematic over- (+) or under- (−) prediction |

> **MBE = 0 does not mean perfect.** Errors of +10 and −10 cancel to MBE 0. MBE measures **bias** (systematic offset), not accuracy — it's the only one of the four that can be negative, and the only one where "0" isn't "perfect."

($\text{MAE}\le\text{RMSE}$ because RMSE is the quadratic mean of $\lvert e_i\rvert$ and MAE the arithmetic mean; they're equal only when every error has the same size.)

---

## Quick Reference

```
ASSESSMENT = evaluate a model's performance        SELECTION = pick its flexibility (Lec 10)
  (* = not on exam: calibration, bootstrap CIs)

SKLEARN LOGREG
  class:   (1/N) Σ loss + λΩ(θ⃗)        sklearn: C Σ loss + Ω(θ⃗)      ⇒ C = 1/(Nλ)
  default penalty='l2', C=1.0, solver='lbfgs'.  L1 → liblinear/saga.  elasticnet → saga ONLY

ACCURACY   (TP+TN)/N  — misleading under class imbalance; assumes equal error costs
           always compare to base case (predict majority class)

CONFUSION MATRIX           Pred −   Pred +
             Actual −       TN       FP   (Type I)
             Actual +       FN       TP   (Type II = FN)
  TPR = sensitivity = recall = TP/(TP+FN)        ÷ actual +
  TNR = specificity          = TN/(TN+FP)        ÷ actual −
  FPR = 1 − specificity      = FP/(TN+FP)        ÷ actual −
  FNR = 1 − sensitivity      = FN/(TP+FN)        ÷ actual +
  precision = PPV            = TP/(TP+FP)        ÷ PREDICTED +
  all in [0,1].  Predict all +: TPR=FPR=1, precision = %pos.   Can't max them all.
  Balanced acc = (TPR + TNR)/2          trivial classifiers → 0.5
  F1 = 2PR/(P+R)  (harmonic; no TN)     Fβ = (1+β²)PR/(β²P + R), β>1 favors recall

THRESHOLDS   default: +1 iff σ(θ⃗·x⃗) > 0.5 ⟺ θ⃗·x⃗ > 0.  new threshold = parallel boundary

ROC    x = FPR, y = TPR, one point per threshold, (0,0) → (1,1), slope ≥ 0
       by hand: sort by score; positive = step UP 1/P, negative = step RIGHT 1/N₋
       diagonal = random;  top-left corner = perfect;  higher non-crossing curve dominates
AUROC  trapezoid rule, sklearn.metrics.auc(x, y) / roc_auc_score
       = P[score(x⁺) > score(x⁻)]  — ranking only, threshold-free
       1 perfect · >0.9 check for LEAKAGE · 0.8 good · 0.5 random · <0.5 wrong
PR     x = recall, y = precision; not monotone.  random baseline AUPRC = % positive
       imbalanced data: ROC hides FPs (TN dilutes FPR); PR exposes them. Report both.
THRESHOLD PICK   fix one requirement (e.g. sensitivity ≥ 0.8), read the others off curves

MULTICLASS  macro = avg of per-class metric (classes equal)
            micro = pool counts (examples equal);  micro-P = micro-R = accuracy

REGRESSION  MSE  [0,∞)  · RMSE [0,∞) same units as y · MAE [0,∞) ≤ RMSE
            MBE (−∞,∞): sign = over/under-prediction; 0 ≠ perfect.  Perfect = 0 for all.
```
