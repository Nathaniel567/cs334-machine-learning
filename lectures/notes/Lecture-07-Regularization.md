# CS-334 Machine Learning — Lecture 07: Regularization
**Date:** 09/17/2026 · **Instructor:** Prof. Shengpu Tang
**Slides:** `lectures/slides/Lecture-07-Regularization.pdf` (includes In-Class Exercise #6)
**Prereq:** `Lecture-06-Linear-Regression.md`

---

## Admin / Logistics

- **HW1 solutions posted.** Grades soon.
- **Office hours — new times:**
  - **Thursday** after class till **3:30pm**
  - **Friday** 10:30–11:30am and 1–1:45pm
  - **Location:** the professor's office, **MSC W302-I**
- **HW2** due **Sunday 9/27, 11:59pm.** Topics: Decision Boundaries (Lec 3), Perceptron (Lec 4), GD/SGD (Lec 5), Linear Regression (Lec 6), **Regularization (Lec 7)**.
- **Extra credit (Weighted Linear Regression)** due **Wed 9/30, 11:59pm.**
- **HW3** will explore the regularizers in §4 further.

---

## 0. Where This Lecture Sits

Lecture 06 ended with a loose end: when $X^TX$ isn't invertible, the pseudoinverse picks the **minimum-norm** solution — it *prefers smaller parameters* when the data can't distinguish between candidates. This lecture turns that preference from an accident of `pinv` into a **deliberate, tunable part of the objective**.

```
Lec 06:  objective = R_N(θ⃗)                 fit the training data as well as possible
         ↓ but "as well as possible" on TRAINING data is the wrong target
Lec 07:  objective = R_N(θ⃗) + λ Ω(θ⃗)        fit the data AND stay simple
```

The lecture has two distinct halves, and it's worth keeping them separate:

1. **The diagnosis** (slides 8–18): why minimizing training error is not the goal at all — generalization, overfitting/underfitting, bias/variance.
2. **The cure** (handwritten slides 20–22): regularization — the objective, the regularizers, ridge regression, and the geometry.

---

## 1. Recap: The Two Tasks and Their Solutions

**Classification** — given $\mathcal{D} = \{\vec{x}^{(i)}, y^{(i)}\}_{i=1}^{N}$ with $\vec{x}^{(i)} \in \mathbb{R}^d$, $y^{(i)} \in \{-1,+1\}$, find a good $h : \mathbb{R}^d \to \{-1,+1\}$:

$$h(\vec{x};\vec{\theta}) = \mathrm{sign}\left(\vec{\theta}\cdot\vec{x}\right)$$

Objective: minimize empirical risk with {zero-one, perceptron, hinge} loss. Three approaches, in the order we met them:

1. **"Eyeballing"** a boundary that minimizes training error
2. **The perceptron algorithm** — *(the slide's parenthetical: **DO NOT USE IRL**)*
3. **(Stochastic) Gradient Descent**

**Regression** — same setup but $y \in \mathbb{R}$, find a good $f : \mathbb{R}^d \to \mathbb{R}$:

$$f(\vec{x};\vec{\theta},b) = \vec{\theta}\cdot\vec{x} + b$$

Objective: minimize empirical risk with **squared error loss**, $\min_{\vec{\theta}} \frac{1}{N}\sum_i \frac{(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)})^2}{2}$. **Two** solutions:

1. **SGD:** $\vec{\theta}^{(k+1)} = \vec{\theta}^{(k)} + \eta_k\left(y^{(i)} - \vec{\theta}^{(k)}\cdot\vec{x}^{(i)}\right)\vec{x}^{(i)}$
2. **Closed form:** $\vec{\theta}^* = \left(X^TX\right)^{-1}X^T\vec{y} = X^{+}\vec{y}$

The slides also show the regression **loss surface** and its **contour plot** for the two-parameter model $f(x;\theta_0,\theta_1) = \theta_0 + \theta_1 x$: a smooth convex bowl, with contours as nested ellipses around a single minimum. Those ellipses come back in §7 — the geometry of regularization is drawn on exactly this picture.

---

## 2. Fitting Non-Linear Functions

> **How can we use linear regression to fit non-linear functions (in two dimensions)?**

The running example (from Bishop, *Pattern Recognition and Machine Learning*) is $N=10$ noisy samples from a sine-like curve on $x \in [0,1]$.

### 2.1 $M = 1$: the straight line

$$\phi(x) = [1, x], \qquad \vec{\theta} = [0.82,\ -1.27]$$

**This doesn't even fit the training data well.** The line cuts through the curve and misses nearly every point.

### 2.2 Polynomial regression: $M = 3$

The trick: **map the data to a higher dimension through an explicit feature mapping.**

$$\phi(x) = [1,\ x,\ x^2,\ x^3], \qquad \vec{\theta} = [0.31,\ 7.99,\ -25.43,\ 17.37]$$

> **The key idea, and it recurs all semester:** the model is **linear in the higher-dimensional space** but **non-linear in the original space.**

Nothing about the algorithm changes. $\phi(x)$ is computed *before* training; the learner still solves an ordinary linear regression, just with $d = 4$ instead of $d = 2$. "Linear model" constrains how $\vec{\theta}$ enters, not how $x$ does.

The $M=3$ fit tracks the underlying curve closely. So — push further?

### 2.3 $M = 9$: where it goes wrong

$$\phi(x) = [1,\ x,\ x^2,\ \dots,\ x^9], \qquad \vec{\theta} = [\theta_0, \theta_1, \dots, \theta_9]$$

With $10$ parameters and $10$ points, the curve passes **through every training point exactly** — training error $\approx 0$.

> **This fits the training data (very) well — so what's wrong?**
>
> Look at the curve *between* the points: it oscillates wildly, swinging far outside the data range to thread each point. Its prediction anywhere new is garbage. And note the coefficients themselves — high-degree fits need **enormous, alternating-sign** weights to produce those swings. **That observation is the whole basis for §3 onward.**

---

## 3. Generalization

> **What we ultimately want is a model that works well on unseen data. But… we've been talking about minimizing empirical risk on training data.** (Recall Lectures #1 & #3.)

Two datasets, two risks — same formula, different data:

| | Data | Risk |
|---|---|---|
| **train** | $\mathcal{D} = \{\vec{x}^{(i)}, y^{(i)}\}_{i=1}^{N}$ | $R(\vec{\theta};\mathcal{D}) = \frac{1}{N}\sum_{i=1}^{N}\mathrm{loss}\left(\vec{x}^{(i)}, y^{(i)}, \vec{\theta}\right)$ |
| **test** | $\mathcal{D}_{\text{test}} = \{\vec{x}^{(i)}, y^{(i)}\}_{i=1}^{N'}$ | $R(\vec{\theta};\mathcal{D}_{\text{test}}) = \frac{1}{N'}\sum_{i=1}^{N'}\mathrm{loss}\left(\vec{x}^{(i)}, y^{(i)}, \vec{\theta}\right)$ |

$$\textbf{Generalization Gap} \quad = \quad R(\vec{\theta};\mathcal{D}_{\text{test}}) - R(\vec{\theta};\mathcal{D})$$

**If we optimize $\vec{\theta}$ by minimizing $R(\vec{\theta};\mathcal{D})$, then $R(\vec{\theta};\mathcal{D}_{\text{test}})$ may be large for two reasons** — and they are *different failures with opposite cures*:

| | Names | |
|---|---|---|
| **Underfitting** | "Bias" · "Approximation Error" | the model class was too weak |
| **Overfitting** | "Variance" · "Estimation Error" | the model class was too rich for this much data |

Each name is a different discipline's vocabulary for the same phenomenon; the professor lists all three because all three show up in papers.

### 3.1 Underfitting / Bias / Approximation Error

The diagnostic device: **imagine training the model on multiple datasets** drawn from the same source.

> **On average, is the prediction of our model close to the true value?**
> *"Error of approximating the real world with a model."*

For $M=1$: every dataset gives roughly the same straight line, and every one of them is wrong in the same way — a line simply cannot be a sine curve. Symptoms:

- **Poor on the training set**
- **Poor on the test set**
- **As $N \uparrow$, doesn't really help.** More data can't fix a model class that doesn't contain the truth.

That last bullet is the sharpest test in this lecture: *if more data wouldn't help, the problem is bias.*

### 3.2 Overfitting / Variance / Estimation Error

> **How much do the predictions of our model trained on different datasets vary (around their average)?**
> *"Are we picking up on the signal or the noise?"*

For $M=9$: each dataset produces a **wildly different** degree-9 curve, because each one chases its own particular noise. Symptoms:

- **(Very) good on the training set**
- **Poor on the test set**
- **As $N \uparrow$, variance $\downarrow$.**

The slides make that last point concretely: the same $M=9$ model fit on $N = 100$ points instead of $10$ is a **good** fit. **The model wasn't wrong — it was underdetermined.** Nine degrees of freedom against ten points is a recipe for chasing noise; against a hundred points it's fine.

> **So "complexity" is never absolute.** It's complexity *relative to the amount of data*. The same $M$ is overfitting at $N=10$ and appropriate at $N=100$.

### 3.3 The Bias–Variance Trade-Off

The classic diagram: as model complexity increases left to right,

```
error
  │  ╲                                    ╱     ← total test error (U-shaped)
  │   ╲  bias² ↓                variance ╱
  │    ╲___                          ___╱
  │        ╲______            ______╱
  │               ╲___  ___╱
  │                   ╲╱  ← the sweet spot
  └──────────────────────────────────────── model complexity
     UNDERFITTING                OVERFITTING
```

Training error decreases **monotonically** with complexity — which is precisely why it is useless as a model-selection criterion. Test error is **U-shaped**. The goal is the bottom of the U, and you cannot see it from the training set alone.

---

## 4. How to Control Model Complexity?

Two solutions, per the slides:

1. **Vary the degree of polynomial features.** Works, but it's **discrete and coarse** — $M \in \{1,2,3,\dots\}$, with no way to ask for "a bit less than $M=9$." It also throws whole features away wholesale.
2. **Regularization.** Continuous, smooth, and it keeps every feature while limiting how hard any of them can be used.

### 4.1 The regularized objective

**Regularization: a "knob" for controlling model complexity.**

$$\boxed{J(\vec{\theta}) = R_N(\vec{\theta}) + \underbrace{\lambda\,\Omega(\vec{\theta})}_{\text{regularization term}}}$$

| Piece | Name | Role |
|---|---|---|
| $R_N(\vec{\theta})$ | empirical risk | fit the data |
| $\Omega(\vec{\theta})$ | **regularizer** / **regularization penalty** | measure of model complexity |
| $\lambda > 0$ | **regularization strength** | **balances how well we fit the data vs. complexity of the model** |

$\lambda$ is the knob, and it is a **hyperparameter** — like $\eta$, it is not learned by the optimizer. At $\lambda \to 0$ we recover plain OLS (maximum overfitting); as $\lambda \to \infty$ the penalty dominates and $\vec\theta \to \vec 0$ (maximum underfitting). Tuning $\lambda$ is *sliding along the horizontal axis of the bias–variance diagram* — which is exactly why this is the answer to §3.

### 4.2 Common regularizers

| | $\Omega(\vec{\theta})$ | Name | Note |
|---|---|---|---|
| **L2** | $\dfrac{\lVert\vec{\theta}\rVert_2^2}{2} = \dfrac{1}{2}\displaystyle\sum_{j=1}^{d}\theta_j^2$ | **"Ridge"** | smooth, shrinks everything |
| **L1** | $\lVert\vec{\theta}\rVert_1 = \displaystyle\sum_{j=1}^{d}\lvert\theta_j\rvert$ | **"LASSO"** | **leads to sparse solutions** |
| **Elastic net** | $\lambda_2\lVert\vec{\theta}\rVert_2^2 + \lambda_1\lVert\vec{\theta}\rVert_1$ | | both at once, two knobs |

*Explore more in HW3.*

"**Sparse**" means many $\theta_j$ are **exactly zero**, not merely small — so L1 performs **feature selection** as a side effect of fitting. §7 explains geometrically why the two behave so differently.

---

## 5. Effect of Regularization

> **When minimizing $J(\vec{\theta})$, we're still trying to minimize $R_N(\vec{\theta})$, but at the same time minimize $\Omega(\vec{\theta})$:**
>
> - **push model parameters toward smaller values**
> - **resist setting parameters away from a default of zero unless the data strongly suggests otherwise**

The second phrasing is the more useful one. Every nonzero weight now has to **pay for itself** in reduced training error. A feature that helps a lot keeps its weight; a feature that helps only by fitting noise gets shrunk away, because the noise-fitting gain is small and the penalty isn't.

> **Why are small parameter values in $\vec{\theta}$ good?**
>
> - **Limited effect of small perturbations of input features on model output.** Since $f = \vec{\theta}\cdot\vec{x}$, a perturbation $\delta$ in feature $j$ moves the output by $\theta_j\delta$. Small $\lVert\vec\theta\rVert$ ⟹ a **stable, smooth** function — and the $M=9$ disaster in §2.3 was precisely a function that was *not* smooth, powered by huge coefficients.
> - **If $\theta_j = 0$ exactly, then that $j$-th feature is effectively unused.** This is the bridge to sparsity: zeroing a weight *is* deleting a feature, so L1 turns complexity control into feature selection.

### 5.1 Occam's Razor

> **William of Occam** (13th-century philosopher):
> *"When you have two competing hypotheses, the simpler one is preferred."*

That's the whole justification, and regularization is just this principle written as arithmetic: $R_N$ measures *how well a hypothesis explains the data*, $\Omega$ measures *how complicated it is*, and $\lambda$ sets the exchange rate.

*(The slide pairs this with the PhD Comics strip contrasting "Occam's Razor" — the simpler of two explanations is likelier true — with "Occam's Professor" — the more complicated of two ways to do something is the one your professor will ask you to do.)*

---

## 6. Ridge Regression — L2-Regularized Linear Regression

> **How does regularization affect our algorithms / solutions for regression?**

$$J(\vec{\theta}) = \sum_{i=1}^{N}\frac{\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)^2}{2} + \underbrace{\lambda\frac{\lVert\vec{\theta}\rVert^2}{2}}_{\text{extra term here}}$$

> **Small caveat here:** the factor of $\frac{1}{N}$ is **"absorbed" into $\lambda$.** Dropping $\frac1N$ from the first term while keeping $\lambda$ symbolic just rescales what $\lambda$ means — the family of solutions traced out as $\lambda$ varies is the same. It keeps the algebra clean, but it does mean **a $\lambda$ that works for one $N$ won't transfer to another.**

### 6.1 SGD for ridge regression

**Let's calculate the gradient:**

$$\nabla_{\vec{\theta}}\left(\frac{\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)^2}{2} + \lambda\frac{\lVert\vec{\theta}\rVert^2}{2}\right) = \left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)\left(-\vec{x}^{(i)}\right) + \lambda\vec{\theta}$$

> **Gradient of the squared L2 norm** (the boxed aside):
> $$\lVert\vec{\theta}\rVert^2 = \theta_1^2 + \dots + \theta_d^2, \qquad \frac{\partial}{\partial\theta_j}\lVert\vec{\theta}\rVert^2 = 2\theta_j$$
> $$\nabla_{\vec{\theta}}\left(\vec{\theta}\cdot\vec{\theta}\right) = \left[\frac{\partial}{\partial\theta_1}\lVert\vec\theta\rVert^2, \dots, \frac{\partial}{\partial\theta_d}\lVert\vec\theta\rVert^2\right]^T = [2\theta_1, 2\theta_2, \dots, 2\theta_d]^T = 2\vec{\theta}$$
> The $\frac{1}{2}$ in the L2 penalty exists exactly to cancel this $2$ — the same bookkeeping trick as in squared loss.

**Update rule:**

$$
\begin{aligned}
\vec{\theta}^{(k+1)} &= \vec{\theta}^{(k)} - \eta_k\left[\left(y^{(i)} - \vec{\theta}^{(k)}\cdot\vec{x}^{(i)}\right)\left(-\vec{x}^{(i)}\right) + \lambda\vec{\theta}^{(k)}\right] \\
&= \boxed{\underbrace{\left(1 - \eta_k\lambda\right)\vec{\theta}^{(k)}}_{\text{shrink parameters toward zero in each update}} + \underbrace{\eta_k\left(y^{(i)} - \vec{\theta}^{(k)}\cdot\vec{x}^{(i)}\right)\vec{x}^{(i)}}_{\text{same as before}}}
\end{aligned}
$$

> **This is the most quotable result in the lecture.** Regularization costs **one multiplication**. The data-driven term is *verbatim* the Lecture 06 update; all L2 adds is that $\vec{\theta}$ gets multiplied by $(1 - \eta_k\lambda) < 1$ **before** it. Every step, the parameters decay a little toward zero, and only the residual term pushes back. This is why L2 regularization is called **"weight decay"** in the neural-network literature — it is literally a decay factor per step.
>
> Note also the constraint it implies: we need $\eta_k\lambda < 1$, or the multiplier goes negative and the parameters flip sign each step.

### 6.2 Closed form

$$J(\vec{\theta}) = \frac{1}{2}\left(X\vec{\theta} - \vec{y}\right)^T\left(X\vec{\theta} - \vec{y}\right) + \frac{\lambda}{2}\vec{\theta}^T\vec{\theta}$$

$$
\begin{aligned}
\nabla_{\vec{\theta}}J(\vec{\theta}) &= \left(X^TX\right)\vec{\theta} - X^T\vec{y} + \lambda\vec{\theta} \\
&= \left(X^TX + \lambda I_d\right)\vec{\theta} - X^T\vec{y} \qquad\qquad (I_d \text{ is the } d\times d \text{ identity matrix})
\end{aligned}
$$

Setting $\left.\nabla_{\vec{\theta}}J(\vec{\theta})\right|_{\vec{\theta}=\vec{\theta}^*} = 0$:

$$\boxed{\vec{\theta}^* = \underbrace{\left(X^TX + \lambda I_d\right)^{-1}}_{\textbf{always invertible}}X^T\vec{y}}$$

> **This closes Lecture 06 §12.** The entire invertibility problem — $d > N$, multicollinearity, duplicated features, non-unique solutions — **disappears.** Adding $\lambda I_d$ shifts every eigenvalue of $X^TX$ up by $\lambda$; since $X^TX$ is positive **semi**-definite (eigenvalues $\ge 0$), the sum has eigenvalues $\ge \lambda > 0$, hence positive definite, hence invertible. **For any $\lambda > 0$ there is exactly one solution, always.**
>
> Compare the three routes to a well-defined answer under multicollinearity: `pinv` gives the minimum-norm solution *among the optimal ones*; ridge changes the objective so that only one solution *is* optimal. The second is a modeling decision with a knob; the first is a numerical fallback. The name "ridge" comes from this $\lambda$ along the diagonal.

---

## 7. Geometric Interpretation of Regularization

$$\min_{\vec{\theta}}\ R_N(\vec{\theta}) + \lambda\Omega(\vec{\theta}) \qquad\underset{\text{by the Lagrange multiplier theorem}}{\Longleftrightarrow}\qquad \min_{\vec{\theta}}\ R_N(\vec{\theta}) \ \ \text{s.t.}\ \ \Omega(\vec{\theta}) \le B$$

$$\textit{unconstrained optimization} \qquad\qquad\qquad\qquad \textit{constrained optimization}$$

These are **two views of the same problem**: penalizing size, or forbidding it past a budget. Each $\lambda$ corresponds to some budget $B$ (and larger $\lambda$ ⟷ smaller $B$). The constrained view is the one you can draw.

> **Where is the optimal solution of the constrained optimization problem?**
>
> **Method of Lagrange multipliers:** the point where the **level curve is tangent with the constraint** is the minimum — **"just touching."**

**Intuition: two forces at play.**

| Force | Effect |
|---|---|
| $\Omega(\vec{\theta}) \le B$ | **shrinks $\vec{\theta}$ to be within the circle / square** |
| $\min R_N(\vec{\theta})$ | **moves $\vec{\theta}$ toward the unconstrained optimum** |

They **settle at a point where the two curves are tangent.**

Why tangency and not crossing: if the constraint boundary *crossed* a level curve of $R_N$, you could slide along the boundary onto a lower level curve — so you weren't at the optimum. Only where they just touch is there nowhere left to slide.

### 7.1 The two pictures, and why L1 is sparse

Both plots show the elliptical contours of $R_N(\vec{\theta})$ in the $(\theta_1,\theta_2)$ plane with the unconstrained optimum $\times$ off in the first quadrant, and the feasible region in blue at the origin:

| | Feasible region | Contact point |
|---|---|---|
| **L2 / Ridge** | $\lVert\vec{\theta}\rVert_2^2 = B$ — a **circle** | a **smooth boundary**, so tangency happens at a generic point: **both** coordinates nonzero, both **smaller** |
| **L1 / LASSO** | $\lVert\vec{\theta}\rVert_1 = B$ — a **diamond** (square rotated 45°) | the diamond has **corners on the axes**, and the corners **stick out toward the contours** — so an expanding ellipse very often touches a **corner first**, where one coordinate is **exactly zero** |

> **That is the whole explanation of sparsity**, and it's worth holding onto as a picture rather than a formula. A circle has no preferred directions, so it never favors putting a weight at exactly zero. A diamond is **pointy at the axes** — and the axes are exactly where coordinates vanish. In $d$ dimensions the L1 ball has corners, edges and faces of every dimension, so the contact point typically has *some* coordinates exactly zero. Hence **feature selection for free.**
>
> It also explains the non-obvious asymmetry: L2 shrinks everything a bit and zeroes nothing; L1 zeroes some things entirely and leaves others nearly untouched.

### 7.2 → In-Class Exercise #6

The slides hand off to the exercise here; work it on the two pictures above.

---

## Quick Reference

```
GENERALIZATION            what we want: low risk on UNSEEN data
  R(θ⃗; D)     = (1/N)  Σᴺ  loss(x⃗⁽ⁱ⁾, y⁽ⁱ⁾, θ⃗)           training
  R(θ⃗; Dtest) = (1/N′) Σᴺ′ loss(x⃗⁽ⁱ⁾, y⁽ⁱ⁾, θ⃗)           test
  generalization gap = R(θ⃗; Dtest) − R(θ⃗; D)

  UNDERFITTING = "bias" = "approximation error"
    poor on train, poor on test;  as N↑ DOESN'T HELP      ← the decisive test
  OVERFITTING  = "variance" = "estimation error"
    (very) good on train, poor on test;  as N↑ variance ↓
  training error falls monotonically with complexity ⇒ useless for model selection
  test error is U-shaped ⇒ aim for the bottom of the U

POLYNOMIAL REGRESSION     explicit feature mapping φ(x), then ordinary lin. reg.
  M=1: φ = [1,x]              underfits — misses the training data
  M=3: φ = [1,x,x²,x³]        good fit
  M=9: φ = [1,x,…,x⁹], N=10   interpolates every point, oscillates between them
       same M=9 with N=100 ⇒ fine.  complexity is RELATIVE TO N
  linear in the higher-dim space, NON-linear in the original space
  huge alternating-sign coefficients are the fingerprint of overfitting

REGULARIZATION            a "knob" for controlling model complexity
  J(θ⃗) = R_N(θ⃗) + λ Ω(θ⃗)        λ > 0 = regularization strength, a HYPERPARAMETER
                                 λ→0: plain OLS (overfit);  λ→∞: θ⃗→0⃗ (underfit)
  L2 "Ridge"    Ω = ‖θ⃗‖²₂ / 2 = ½ Σ θⱼ²
  L1 "LASSO"    Ω = ‖θ⃗‖₁ = Σ |θⱼ|              ⇒ SPARSE solutions
  elastic net   Ω = λ₂‖θ⃗‖²₂ + λ₁‖θ⃗‖₁
  why small θ⃗: small input perturbations ⇒ small output change (smooth model);
               θⱼ = 0 exactly ⇒ feature j effectively unused
  Occam's Razor: of two competing hypotheses, prefer the simpler

RIDGE REGRESSION          J(θ⃗) = Σ (y⁽ⁱ⁾ − θ⃗·x⃗⁽ⁱ⁾)²/2 + λ‖θ⃗‖²/2   (1/N absorbed into λ)
  ∇_θ ‖θ⃗‖² = 2θ⃗                                 (the ½ cancels it)
  SGD:  θ⃗⁽ᵏ⁺¹⁾ = (1 − η_k λ) θ⃗⁽ᵏ⁾ + η_k (y⁽ⁱ⁾ − θ⃗⁽ᵏ⁾·x⃗⁽ⁱ⁾) x⃗⁽ⁱ⁾
        └ shrink toward 0 each update ┘  └──── same as Lecture 06 ────┘
        = "weight decay".  needs η_k λ < 1.
  closed form:  ∇J = (XᵀX + λI_d)θ⃗ − Xᵀy⃗ = 0
        θ⃗* = (XᵀX + λI_d)⁻¹ Xᵀy⃗       ALWAYS INVERTIBLE for λ > 0
        (eigenvalues of XᵀX ≥ 0, shifted up by λ ⇒ positive definite)
        ⇒ fixes every Lec 06 §12 failure: d > N, multicollinearity, non-uniqueness

GEOMETRY
  min R_N + λΩ(θ⃗)   ⟺   min R_N s.t. Ω(θ⃗) ≤ B      (Lagrange multiplier theorem)
  unconstrained          constrained;  larger λ ⟷ smaller B
  optimum = where a level curve of R_N is TANGENT to the constraint ("just touching")
  two forces: Ω ≤ B shrinks θ⃗ inside the region; min R_N pulls toward the
              unconstrained optimum ⇒ settle where the curves are tangent
  L2 ⇒ CIRCLE ‖θ⃗‖²₂ = B : smooth boundary, generic contact, nothing exactly zero
  L1 ⇒ DIAMOND ‖θ⃗‖₁ = B : CORNERS ON THE AXES, contact at a corner
                           ⇒ a coordinate is exactly 0 ⇒ sparsity / feature selection
```
