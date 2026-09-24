# CS-334 Machine Learning — Lecture 08: Logistic Regression
**Date:** 09/22/2026 · **Instructor:** Prof. Shengpu Tang
**Slides:** `lectures/slides/Lecture-08-Logistic-Regression.pdf` (includes In-Class Exercise #7)
**Prereq:** `Lecture-07-Regularization.md`

---

## Admin / Logistics

- **HW1 grades released.** Solutions are on Canvas; graded homework is on Gradescope. **Submit a regrade request** if you believe there's an error.
- **HW2** due **Sunday 9/27, 11:59pm.** Topics: Decision Boundaries (Lec 3), Perceptron (Lec 4), GD/SGD (Lec 5), Linear Regression (Lec 6), Regularization (Lec 7).
- **Extra credit (Weighted Linear Regression)** due **Wed 9/30, 11:59pm.**

---

## 0. Where This Lecture Sits

The first two-thirds of the deck (slides 5–25) is a **recap** of Lecture 07 plus the sklearn API. The new material is the last third — slides 26–27 and the handwritten pages 28–30.

```
Lec 03–05:  classification   y ∈ {−1,+1}   h = sign(θ⃗·x⃗)     outputs a DECISION
Lec 06–07:  regression       y ∈ ℝ         f = θ⃗·x⃗          outputs a VALUE
Lec 08:     logistic reg.    y ∈ {−1,+1}   h = σ(θ⃗·x⃗)       outputs a PROBABILITY
```

Despite the name, **logistic regression is a classifier.** It trains on binary labels; what's "regression" about it is that its output is a continuous number in $[0,1]$ rather than a hard $\pm1$.

The lecture's arc is the same as Lecture 05's, and it's worth seeing that:

1. pick a model ($\sigma(\vec\theta\cdot\vec x)$),
2. turn "what we want" into an **empirical risk** (here via maximum likelihood — a new route to a loss),
3. differentiate it and run **SGD**.

---

## 1. Recap: Bias–Variance

### 1.1 The dartboard picture

Imagine training on many datasets; each dot is one trained model's prediction, the bullseye is the truth.

| | Low variance | High variance |
|---|---|---|
| **Low bias** | tight cluster on the bullseye — the goal | scattered, but centred on the bullseye |
| **High bias** | tight cluster, **off**-centre | scattered **and** off-centre |

- **Bias** = how far the *average* dot is from the bullseye.
- **Variance** = how spread out the dots are around their own average.

Underfitting is the high-bias / low-variance board; overfitting is the low-bias / high-variance board.

### 1.2 The error curves

Two handwritten plots, both against **model complexity**:

- **Left:** bias falls, variance rises, **total error = their sum is U-shaped.**
- **Right:** **training error** falls monotonically; **test error ("generalization error")** is U-shaped.

The summary slide puts them together: *high bias / low variance* to the left, *high variance / low bias* to the right, and a dashed line at the **optimal model complexity** at the bottom of the testing-error U.

### 1.3 Definitions

- **Generalization** — performance of a model on independent / future unseen data (data *not* used in training).
- **Underfitting** — the model can't capture the input–output relationship; **high error on both training and test data.**
- **Overfitting** — the model is specific to the training set and learns the **noise** rather than a generalizable rule; **low training error, high test error.**

The slides (sklearn's polynomial-degree example, degrees 1 / 4 / 15) label the same three fits four ways, and all four vocabularies are worth recognizing:

| Degree 1 | Degree 4 | Degree 15 |
|---|---|---|
| poor on training set, poor at predicting | **generalizable** | very good on training set, poor at predicting |
| underfitting | **just right** | overfitting |
| high bias, low variance | **low bias, low variance** | low bias, high variance |
| high approximation error, low estimation error | **low, low** | low approximation error, high estimation error |

**Classification version (slide 14):** same story with decision boundaries — a straight line through two interleaved classes (underfit), a smooth curve (just right), and a boundary that snakes around every individual point (overfit).

### 1.4 Sources and cures — this is the new part of the recap

| | **Bias** | **Variance** |
|---|---|---|
| **Sources** | model class **too small** — can't represent the underlying relationship; models **"too global"** (e.g. constant output, a single linear separator) | **noise** in labels or features; models **"too local"** — sensitive to small changes in feature values; training set **too small** |
| **Cure** | **more complex models** | **more data**, or **less complex models** |
| **How** | interaction terms · polynomial features · collect more features · kernels (Lec 11) · algorithm-specific approaches | drop interaction terms · **regularization** · feature selection (Lec 11) · don't use kernels (Lec 11) · ensembles (Lec 14) |

> **The two columns are mirror images.** Nearly every bias cure appears, negated, as a variance cure (add interactions ↔ drop them; use kernels ↔ don't). That's the trade-off in table form: any knob that moves you one way costs you on the other. The one exception is **more data**, which cuts variance *without* adding bias — which is why it's always the first thing to try when it's available.

"Too global" vs "too local" is a useful mental model: a global model makes one decision for the whole input space and can't bend; a local model reacts to each nearby point and bends too much.

---

## 2. Recap: Regularization

$$J(\vec\theta) = \underbrace{R_N(\vec\theta)}_{\substack{\text{empirical risk /}\\ \text{proxy for training error}}} + \underbrace{\lambda\,\Omega(\vec\theta)}_{\text{regularization term / penalty}} \qquad\text{(regularized empirical risk)}$$

$\lambda$ is the **"knob"** that controls how much I care about **(1) fitting the data** vs **(2) reducing model complexity**.

> **"If I make $\lambda$ larger, I'm paying more attention to…"** — **(2), reducing model complexity.** Larger $\lambda$ ⟹ smaller $\vec\theta$ ⟹ more bias, less variance. (Equivalently, in the constrained view below, a smaller budget $B$.)

$$\min_{\vec\theta}\ R_N(\vec\theta) + \lambda\Omega(\vec\theta) \quad\Longleftrightarrow\quad \min_{\vec\theta}\ R_N(\vec\theta)\ \ \text{s.t.}\ \ \Omega(\vec\theta)\le B$$

Intuition, unchanged from Lecture 07: push parameters toward a default of 0; resist moving them unless the data strongly suggests otherwise; Occam's Razor.

### 2.1 The three regularizers, compared

Slide 20 is the Lecture 07 geometry with labels: **L1 norm → "sparsity inducing"** (diamond, contact at a corner on the $\theta_2$ axis), **L2 norm → "weight sharing"** (circle, contact off-axis), **L1 + L2 → "compromise… two parameters"** (a rounded diamond).

| Penalty | Effect | Solving it |
|---|---|---|
| $\lVert\vec\theta\rVert_2^2 = \sum_j \theta_j^2$ | encourages **small** coefficients (but not necessarily 0). Also called **shrinkage** or **weight decay** | **closed form exists** |
| $\lVert\vec\theta\rVert_1 = \sum_j \lvert\theta_j\rvert$ | drives some coefficients to **zero** — **sparsity**, effectively selecting features | **no closed form**, but efficient algorithms exist (variants of SGD) |
| $\lambda_2\lVert\vec\theta\rVert_2^2 + \lambda_1\lVert\vec\theta\rVert_1$ | elastic net **balances** the two: selects features like LASSO, shrinks coefficients of **correlated** predictors like ridge | |

Why LASSO has no closed form: $\lvert\theta_j\rvert$ isn't differentiable at $0$ — exactly the point LASSO wants to land on — so you can't just set a gradient to zero and solve.

"Weight sharing" for L2 is worth one sentence: given two perfectly correlated features, the squared penalty is minimized by **splitting** the weight evenly between them ($a^2+b^2$ with $a+b$ fixed is smallest at $a=b$), whereas L1 is indifferent to the split and tends to pick one. That's the "shrinks correlated predictors like ridge" bullet.

### 2.2 Practical notes

1. **The offset / intercept is usually left unpenalized** (depending on the implementation):
   $$\min_{\vec\theta,\,b}\ \sum_{i=1}^{N}\frac{\left(\vec\theta\cdot\vec x^{(i)} + b - y^{(i)}\right)^2}{2} + \lambda\frac{\lVert\vec\theta\rVert^2}{2}$$
   $b$ appears in the loss (circled on the slide) but **not** in the penalty. Shrinking $b$ toward 0 would just bias every prediction toward 0 — it controls *where* the function sits, not how complex it is.
2. **Regularization strength can be "unfair" if features are on different scales.** Example: $x_1$ = age in **years**, $x_2$ = weight in **grams**. Weight in grams has huge values, so it needs a tiny $\theta_2$ — which the penalty barely notices — while age needs a much larger $\theta_1$ that gets penalized hard. The *same* $\lambda$ regularizes the two features very differently for no reason other than units.
3. **So feature preprocessing (centering + normalization) is important** → HW3.

---

## 3. sklearn Implementations

**Regression** (`sklearn.linear_model`):

| Class | Notes |
|---|---|
| `LinearRegression` | wrapper of `scipy.linalg.lstsq`, uses **SVD** |
| `Ridge` | can select the solver (SGD variants, SVD, etc.) |
| `Lasso` | **`Lasso` is just `ElasticNet` with the L2 penalty "turned off"**; implemented as **coordinate descent** |
| `ElasticNet` | |
| `SGDRegressor` | generic SGD routine — mix and match loss and regularizer |

**Linear classification:**

| Class | Notes |
|---|---|
| `linear_model.LogisticRegression` | L1 / L2 / elastic-net regularized logistic regression. **`C = 1/λ` is the *inverse* of regularization strength** |
| `linear_model.SGDClassifier` | mix and match loss and regularizer |
| `linear_model.Perceptron` | SGD with perceptron loss and no regularization |
| `svm.LinearSVC`, `svm.SVC` | Support Vector Machines — hinge loss; `SVC` supports kernels |

> **The `C` trap.** In `LogisticRegression` and the SVMs, **larger `C` means *less* regularization** — the opposite direction from $\lambda$. Everywhere else in this course, "turn up the knob" means more regularization.

**Every sklearn model is the same two lines,** and it maps directly onto the course's notation:

```python
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)      # θ⃗*, b* = argmin_{θ⃗,b}  R_N(θ⃗) + λΩ(θ⃗)
y_pred = clf.predict(X_test)   # h(x⃗; θ⃗, b) = σ(θ⃗·x⃗ + b)   ← some output function of θ⃗·x⃗ + b
```

`.fit` **is** the optimization problem; `.predict` **is** the model. Choosing a class is choosing a loss, a regularizer, and a solver.

---

## 4. Motivation: Predicting Probabilities

| Classification: binary decisions | Regression: predict values |
|---|---|
| $y \in \{-1,+1\}$ | $y \in \mathbb{R}$ |
| approve/decline mortgage? · cat or dog? · will it rain tomorrow? | price of a house · value of a stock tomorrow · highest temperature tomorrow |

> **Some applications are not properly captured with either setup:**
> - probability of a heart attack given medical conditions
> - probability of defaulting on a loan
> - probability of recidivism / reoffending
>
> $y \in \{-1,+1\}$, **but we want to predict** $\Pr[y=+1 \mid \vec x] \in [0,1]$.

Why neither old tool works:

- **Classification** gives only $\mathrm{sign}(\vec\theta\cdot\vec x)$ — a decision with no confidence attached. "Defaults" and "barely defaults" look identical.
- **Regression** on the $\pm1$ labels outputs any real number — $1.7$ or $-3.2$ aren't probabilities.

The labels we *observe* are binary; the quantity we *want* is continuous. Logistic regression fits the second while training on the first.

---

## 5. The Sigmoid Function

> **How do we make a linear model output a probability?** We already have $\vec\theta\cdot\vec x \in \mathbb{R}$.
>
> → **Apply a squashing function** $\sigma: \mathbb{R}\to[0,1]$.

$$\boxed{\sigma(z) = \frac{1}{1+e^{-z}}} \qquad\text{the sigmoid function}$$

An S-curve in $z = \vec\theta\cdot\vec x$, flat near 0 on the far left, flat near 1 on the far right, passing through $0.5$ at $z=0$.

**Range of the sigmoid:**

| | |
|---|---|
| $z\to+\infty$ | $\sigma(z)\to 1$ |
| $z\to-\infty$ | $\sigma(z)\to 0$ |
| $z = 0$ | $\sigma(z) = 0.5$ |

**Useful properties:**

$$\sigma(-z) = 1-\sigma(z) \qquad\qquad \frac{d}{dz}\sigma(z) = \sigma(z)\left(1-\sigma(z)\right)$$

- ✓ continuous
- ✓ differentiable
- ✱ **the sigmoid function is *not* convex** (it's convex for $z<0$ and concave for $z>0$). Remember this — it's why §7 has to work to get a convex loss rather than just using $\sigma$ directly.

Both identities are one line each, and worth being able to reproduce:

- $1-\sigma(z) = \dfrac{e^{-z}}{1+e^{-z}} = \dfrac{1}{e^{z}+1} = \sigma(-z)$ (multiply top and bottom by $e^{z}$).
- $\sigma'(z) = \dfrac{e^{-z}}{(1+e^{-z})^2} = \underbrace{\dfrac{1}{1+e^{-z}}}_{\sigma(z)}\cdot\underbrace{\dfrac{e^{-z}}{1+e^{-z}}}_{1-\sigma(z)}$.

---

## 6. The Logistic Regression Model

$$\boxed{h(\vec x;\vec\theta) = \sigma(\vec\theta\cdot\vec x) = \frac{1}{1+e^{-\vec\theta\cdot\vec x}}}$$

**What we want:** $\sigma(\vec\theta\cdot\vec x)$ should *be* the probability of the positive class,

$$\Pr[y=+1 \mid \vec x] = \sigma(\vec\theta\cdot\vec x).$$

**It's still a linear classifier.** $\sigma$ is monotonic and crosses $0.5$ exactly at $z=0$, so thresholding at $0.5$ gives

$$\sigma(\vec\theta\cdot\vec x) \ge 0.5 \iff \vec\theta\cdot\vec x \ge 0,$$

the **same hyperplane** $\vec\theta\cdot\vec x = 0$ as Lecture 03. What's new is the reading of *distance from the boundary*: $\lvert\vec\theta\cdot\vec x\rvert$ large ⟹ probability near 0 or 1 (confident); near the boundary ⟹ near $0.5$ (unsure). In Lecture 03 the length of $\vec\theta$ didn't matter; **now it does** — scaling $\vec\theta$ up makes every prediction more extreme. That fact is the whole of §9.2.

> **How do we train this classifier? What loss function should we use?**

---

## 7. From Likelihood to Logistic Loss

**Idea:** if $\sigma(\vec\theta\cdot\vec x)$ truly is the probability, then we can use it to write down the **"likelihood"** of the training data $\{\vec x^{(i)}, y^{(i)}\}_{i=1}^{N}$ — and choose the $\vec\theta$ that makes the observed labels most probable.

### 7.1 One example

For each example $\vec x^{(i)}$, the probability of seeing its label being $y^{(i)}$ is

$$
\Pr\left[y=y^{(i)} \mid \vec x^{(i)};\vec\theta\right] =
\begin{cases}
\sigma(\vec\theta\cdot\vec x^{(i)}) & \text{if } y^{(i)}=+1 \\
1-\sigma(\vec\theta\cdot\vec x^{(i)}) = \sigma(-\vec\theta\cdot\vec x^{(i)}) & \text{if } y^{(i)}=-1
\end{cases}
\quad\Longrightarrow\quad \boxed{\sigma\!\left(y^{(i)}\,\vec\theta\cdot\vec x^{(i)}\right)}
$$

> **This is where $y\in\{-1,+1\}$ pays off.** Using $\sigma(-z) = 1-\sigma(z)$, the label just flips the sign of the argument, and both cases collapse into one expression. And that argument, $y^{(i)}(\vec\theta\cdot\vec x^{(i)})$, is the **margin $z$ from Lecture 05** — so the probability of the correct label is simply $\sigma(\text{margin})$.

### 7.2 The whole training set

Since each training example is generated **independently**, probabilities multiply:

$$\Pr\left[\{y^{(i)}\}_{i=1}^{N} \,\middle|\, \{\vec x^{(i)}\}_{i=1}^{N};\vec\theta\right] = \prod_{i=1}^{N}\frac{1}{1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}}$$

**Goal: maximize the likelihood of the training data.**

$$\vec\theta^* = \arg\max_{\vec\theta}\ \prod_{i=1}^{N}\frac{1}{1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}}$$

**Problems** (the professor's red annotations): *we normally **minimize**, not maximize*, and *the **product** looks annoying.* We want something in the familiar empirical-risk form

$$\vec\theta^* = \arg\min_{\vec\theta}\ \sum_{i=1}^{N}\mathrm{loss}\left(\vec x^{(i)}, y^{(i)}, \vec\theta\right).$$

### 7.3 Two tricks

**① Take the log** — product → sum. Since $\log(x)$ is increasing for $x>0$, $\max f(x)$ is equivalent to $\max \log f(x)$ (same $\arg\max$):

$$\arg\max_{\vec\theta}\ \log\prod_{i=1}^{N}\frac{1}{1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}} = \arg\max_{\vec\theta}\ \sum_{i=1}^{N}\log\left(\frac{1}{1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}}\right)$$

**② Take the negative** — $\arg\max_x f(x) = \arg\min_x\left(-f(x)\right)$:

$$\cdots = \arg\min_{\vec\theta}\ \sum_{i=1}^{N}-\log\left(\frac{1}{1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}}\right) = \arg\min_{\vec\theta}\ \sum_{i=1}^{N}\log\left(1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}\right)$$

(using $-\log(1/a) = \log a$).

The log also matters numerically, not just for convenience: a product of $N$ numbers in $(0,1)$ underflows to $0.0$ in floating point for even modest $N$; a sum of logs doesn't.

### 7.4 Logistic loss

Now we can view this as an **empirical risk**:

$$\boxed{\mathrm{loss}_{\log}\left(\vec x^{(i)}, y^{(i)}, \vec\theta\right) = \log\left(1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}\right)} \qquad \text{i.e. } \mathrm{loss}_{\log}(z) = \log(1+e^{-z}),\ \ z = y\,\vec\theta\cdot\vec x$$

- ✓ continuous
- ✓ differentiable
- ✓ **convex**

Plotted against the margin $z$: decreasing, passes through **$\log 2$ at $z=0$**, approaches 0 as $z\to\infty$, and grows **roughly linearly** as $z\to-\infty$.

> **Put it next to the losses from Lecture 05**, all as functions of the same margin $z = y(\vec\theta\cdot\vec x)$:
>
> | Loss | $\mathrm{loss}(z)$ | Shape |
> |---|---|---|
> | zero-one | $\mathbb{1}[z\le 0]$ | step — not continuous, not convex |
> | hinge | $\max\{0,\,1-z\}$ | convex, kink at $z=1$, **exactly 0** for $z\ge1$ |
> | **logistic** | $\log(1+e^{-z})$ | convex, **smooth everywhere**, **never exactly 0** |
>
> Logistic loss is a **smoothed hinge**: both are ~linear for very wrong points and ~0 for very right ones. The difference is that logistic loss *never stops rewarding* a larger margin — which is the seed of the pathological case in §9.2.
>
> And note what we got for free: we didn't *design* this loss the way hinge loss was designed in Lecture 05. It **fell out** of asking "which $\vec\theta$ makes the data most probable?" That's **maximum likelihood estimation**, and it's a general recipe for turning a probabilistic model into a loss.

---

## 8. SGD for Logistic Regression

> **Let's derive the SGD update rule!**

The gradient of one example's loss (the boxed derivation), using $\frac{d}{dx}\log(x) = \frac1x$ and $\frac{d}{dx}e^x = e^x$ with the chain rule:

$$
\begin{aligned}
\nabla_{\vec\theta}\log\left(1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}\right)
&= \frac{1}{1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}}\ \nabla_{\vec\theta}\left(1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}\right) \\
&= \frac{1}{1+e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}}\left(e^{-y^{(i)}\vec\theta\cdot\vec x^{(i)}}\right)\nabla_{\vec\theta}\left(-y^{(i)}\vec\theta\cdot\vec x^{(i)}\right) \\
&= \frac{1}{e^{y^{(i)}\vec\theta\cdot\vec x^{(i)}}+1}\left(-y^{(i)}\vec x^{(i)}\right) \\
&= -y^{(i)}\vec x^{(i)}\,\sigma\!\left(-y^{(i)}\vec\theta\cdot\vec x^{(i)}\right) = \boxed{-y^{(i)}\vec x^{(i)}\left(1-\sigma\!\left(y^{(i)}\vec\theta\cdot\vec x^{(i)}\right)\right)}
\end{aligned}
$$

(Step 2→3: divide top and bottom by $e^{-y\vec\theta\cdot\vec x}$. Step 3→4: $\frac{1}{1+e^{a}} = \sigma(-a)$, then $\sigma(-z) = 1-\sigma(z)$.)

**Therefore, the SGD update rule is**

$$\vec\theta^{(k+1)} = \vec\theta^{(k)} - \eta_k\left[-y^{(i)}\vec x^{(i)}\left(1-\sigma\!\left(y^{(i)}\vec\theta^{(k)}\cdot\vec x^{(i)}\right)\right)\right]$$

$$\boxed{\vec\theta^{(k+1)} = \vec\theta^{(k)} + \eta_k\,\underbrace{\left(1-\sigma\!\left(y^{(i)}\vec\theta^{(k)}\cdot\vec x^{(i)}\right)\right)}_{\Pr[\text{wrong label}]\ \in\ (0,1)}\,y^{(i)}\vec x^{(i)}}$$

> **Read it as a soft perceptron.** The perceptron (Lecture 04) updates $\vec\theta \mathrel{+}= y^{(i)}\vec x^{(i)}$ **only** when the example is misclassified, and by a full step. Logistic regression moves in the *same direction* $y^{(i)}\vec x^{(i)}$ on **every** example, scaled by $1-\sigma(\text{margin})$ — the probability the model currently assigns to the *wrong* label:
>
> - confidently right (margin $\gg 0$): factor $\approx 0$, barely moves
> - on the boundary (margin $= 0$): factor $= 0.5$
> - confidently wrong (margin $\ll 0$): factor $\approx 1$, a full perceptron-sized step
>
> The perceptron's 0/1 "was I wrong?" becomes a continuous "*how* wrong was I?" — which is exactly what replacing the zero-one loss with a smooth one should buy.

---

## 9. Solutions — and When They Misbehave

- **Unfortunately, there is no known closed-form solution for the general case.** Unlike Lecture 06, setting the gradient to zero gives equations with $\sigma(\cdot)$ inside a sum, which can't be solved for $\vec\theta$ algebraically. **(S)GD is the method.**
- **Since the empirical risk is strictly convex**\*, **there is a unique global minimum** — so (S)GD with a suitable step size finds *the* answer, not *an* answer.

Two footnoted pathological cases (red on the slide):

### 9.1 ✱1 — linearly dependent features

**Unless there are linearly dependent features**, in which case there are **infinitely many minima that are equally good.** This is the logistic-regression twin of Lecture 06's non-invertible $X^TX$: if feature 3 = feature 1 + feature 2, weight can be traded among them without changing any $\vec\theta\cdot\vec x$, so the loss is flat along that direction — convex, but not *strictly*.

### 9.2 ✱2 — linearly separable data

**What we *want* logistic regression to output:** a 2-D example with $\times$ (positive) and $\circ$ (negative) points and a boundary $\vec\theta\cdot\vec x = 0$ along the diagonal. Points far from the boundary get confident probabilities (80–90%), points near it get unsure ones (40–60%); on the sigmoid plot they spread along the gentle middle of the S-curve.

**However, if the data is linearly separable and we maximize likelihood,** logistic regression predicts **100% for every positive and 0% for every negative.** The S-curve collapses into a step: every point sits at a large $\lvert z\rvert$, with $\sigma(z)\to 0$ or $\sigma(z)\to 1$.

> **→ In-Class Exercise #7.** (The answers are faint on the slide, so work it first:)
>
> - **How can this happen?** When $\lVert\vec\theta\rVert\to\infty$.
> - **Solution?** **Regularization.**
>
> Why: once some $\vec\theta$ separates the data, every margin is positive, and doubling $\vec\theta$ doubles every margin and **strictly lowers every loss term** — logistic loss is never exactly zero (§7.4), so there's always a bit more to gain. The minimum is never reached; SGD pushes $\lVert\vec\theta\rVert$ upward forever and the probabilities become meaningless 0s and 1s — a classic overfit. Adding $\lambda\lVert\vec\theta\rVert^2/2$ makes that growth cost something, so the regularized objective has a finite minimizer again. **This is why the recap in §2–3 is in this lecture**, and why sklearn's `LogisticRegression` is regularized **by default** (`C=1.0`).
>
> Compare hinge loss: it hits exactly 0 once every margin is $\ge 1$, so it has no incentive to keep growing $\vec\theta$ — the difference traced back to one row of the §7.4 table.

---

## Quick Reference

```
BIAS–VARIANCE (recap)
  bias     = avg prediction far from truth      variance = predictions spread around their avg
  underfit = high bias / low var  = high approx. error  — poor on train AND test
  overfit  = low bias / high var  = high estim. error   — good on train, poor on test
  total error = bias + variance  → U-shaped;   training error ↓ monotonically
  BIAS sources: model class too small, "too global"
       cures:   more complex — interactions, poly features, more features, kernels
  VAR  sources: noisy labels/features, "too local", training set too small
       cures:   MORE DATA, or less complex — drop interactions, REGULARIZATION,
                feature selection, no kernels, ensembles

REGULARIZATION (recap)
  J(θ⃗) = R_N(θ⃗) + λΩ(θ⃗)      λ↑ ⇒ care more about reducing complexity
  L2  ‖θ⃗‖²₂   small coefs, not 0 — "shrinkage"/"weight decay"/"weight sharing"; closed form
  L1  ‖θ⃗‖₁    some coefs exactly 0 — sparsity/feature selection; NO closed form
  elastic net   selects like LASSO, shrinks correlated predictors like ridge
  intercept b usually NOT penalized
  different feature scales ⇒ unfair penalty ⇒ center + normalize (HW3)

SKLEARN
  LinearRegression (lstsq/SVD) · Ridge · Lasso = ElasticNet w/o L2 (coord. descent)
  SGDRegressor / SGDClassifier: mix & match loss + regularizer
  Perceptron = SGD, perceptron loss, no reg. · LinearSVC / SVC: hinge loss
  LogisticRegression:  C = 1/λ   ← LARGER C = LESS regularization. Regularized by default.
  .fit ⇔ argmin R_N + λΩ         .predict ⇔ h(x⃗; θ⃗, b)

SIGMOID                σ(z) = 1 / (1 + e⁻ᶻ)     ℝ → [0,1]
  z→+∞: σ→1   z→−∞: σ→0   σ(0) = 0.5
  σ(−z) = 1 − σ(z)        σ′(z) = σ(z)(1 − σ(z))
  continuous, differentiable, NOT convex

LOGISTIC REGRESSION    y ∈ {−1,+1}, want Pr[y=+1 | x⃗] ∈ [0,1]  — a CLASSIFIER
  h(x⃗; θ⃗) = σ(θ⃗·x⃗)       σ ≥ 0.5 ⟺ θ⃗·x⃗ ≥ 0 ⇒ still a linear boundary
  ‖θ⃗‖ now MATTERS: bigger θ⃗ ⇒ more extreme probabilities
  Pr[y⁽ⁱ⁾ | x⃗⁽ⁱ⁾; θ⃗] = σ(y⁽ⁱ⁾ θ⃗·x⃗⁽ⁱ⁾)            = σ(margin)
  likelihood = Π σ(y⁽ⁱ⁾ θ⃗·x⃗⁽ⁱ⁾)                   (examples independent)
  max likelihood → log (Π→Σ) → negate (max→min):
    θ⃗* = argmin Σ log(1 + e^{−y⁽ⁱ⁾ θ⃗·x⃗⁽ⁱ⁾})

LOGISTIC LOSS          loss_log(z) = log(1 + e⁻ᶻ),  z = y θ⃗·x⃗
  continuous, differentiable, CONVEX;  = log 2 at z = 0;  never exactly 0
  ≈ smoothed hinge.   zero-one 1[z≤0] · hinge max{0,1−z} · logistic log(1+e⁻ᶻ)

SGD
  ∇ = −y⁽ⁱ⁾x⃗⁽ⁱ⁾ σ(−y⁽ⁱ⁾θ⃗·x⃗⁽ⁱ⁾) = −y⁽ⁱ⁾x⃗⁽ⁱ⁾ (1 − σ(y⁽ⁱ⁾θ⃗·x⃗⁽ⁱ⁾))
  θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ + η_k (1 − σ(y⁽ⁱ⁾ θ⃗⁽ᵏ⁾·x⃗⁽ⁱ⁾)) y⁽ⁱ⁾ x⃗⁽ⁱ⁾
            └─ Pr[wrong label] ─┘ × perceptron direction   = "soft perceptron"

SOLUTIONS
  NO closed form — use (S)GD
  strictly convex ⇒ unique global minimum, EXCEPT:
   ✱1 linearly dependent features ⇒ infinitely many equally good minima
   ✱2 linearly separable data ⇒ ‖θ⃗‖ → ∞, predicts 100% / 0%, no finite minimizer
      FIX: REGULARIZATION
```
