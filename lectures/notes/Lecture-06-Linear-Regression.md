# CS-334 Machine Learning — Lecture 06: Linear Regression
**Date:** 09/15/2026 · **Instructor:** Prof. Shengpu Tang
**Slides:** `lectures/slides/Lecture-06-Linear-Regression.pdf` (includes In-Class Exercise #5)
**Prereq:** `Lecture-05-Gradient-Descent.md`

---

## Admin / Logistics

- **HW1** was due **this past Sun 11:59pm**. Solutions posted **tomorrow at 11:59pm**; grades **early next week**.
  - Topics it covered (Prereqs + Lec 1&2): review of vectors & planes, review of linear algebra, `numpy` and `pandas`, data exploration, plotting in Python.
- **HW2 released** — due **in two weeks, Sun 11:59pm**. Topics: Decision Boundaries (Lec 3), Perceptron (Lec 4), GD/SGD (Lec 5), **Linear Regression (Lec 6)**, **Regularization (Lec 7)**.
- **Extra credit: Weighted Linear Regression** — due **3 days after the HW2 deadline**.
- Course resources, restated: lecture slides and notes, textbook readings (**optional**), **Panopto** recording.

---

## 0. Where This Lecture Sits

Everything through Lecture 05 was **classification** — $y$ lives in a finite set, and the model outputs a side of a hyperplane. This lecture changes the *output type* and asks what survives.

```
Lec 03–05:  y ∈ {−1,+1}    classification   → 0-1 / perceptron / hinge loss
Lec 06:     y ∈ ℝ          REGRESSION       → squared loss
                                             → SGD  (same machinery as Lec 05)
                                             → closed form (new: solve it exactly)
```

The answer is that **most of it survives**. Empirical risk, gradients, SGD, learning rates, convergence criteria — all unchanged. Only the loss function changes. And because the new loss is *quadratic*, something genuinely new becomes available: we can set the gradient to zero and solve **analytically**, no iteration at all.

The first half of the lecture is a long recap of Lecture 05 (slides 4–18); the regression material starts at slide 19 and lives mostly in the handwritten pages 22–25.

---

## 1. Recap: Empirical Risk and Hinge Loss

$$R_N(\vec{\theta}) = \frac{1}{N}\sum_{i=1}^{N}\mathrm{loss}\left(y^{(i)}, \vec{\theta}\cdot\vec{x}^{(i)}\right)$$

With hinge loss:

$$R_N(\vec{\theta}) = \frac{1}{N}\sum_{i=1}^{N}\max\left\{0,\ 1 - y^{(i)}\,\vec{\theta}\cdot\vec{x}^{(i)}\right\}$$

> **Why hinge loss instead of zero-one loss?** Because zero-one loss is discontinuous, non-differentiable and non-convex — its **loss surface** over $\vec{\theta}$ is a staircase of flat plateaus, and minimizing it directly is NP-hard. Hinge loss is continuous and convex, so a gradient method has something to descend. (Full argument in Lecture 05 §4–5.)

The slides show the three **loss surfaces** side by side — the same picture is worth re-reading:

| Loss | Surface over $\vec{\theta}$ |
|---|---|
| **zero-one** | flat terraces with vertical cliffs — no usable gradient anywhere |
| **perceptron** $\max\{0,-z\}$ | piecewise linear, convex, but flat for all $z>0$ — zero gradient as soon as a point is correct |
| **hinge** $\max\{0,1-z\}$ | piecewise linear, convex, and still sloped for $0 < z < 1$ — keeps pushing for **margin** |

---

## 2. Recap: Gradients, GD, and SGD

Let $f(\vec{\theta}) : \mathbb{R}^d \to \mathbb{R}$; we want $\vec{\theta}^*$ minimizing it.

$$\nabla_{\vec{\theta}}\, f(\vec{\theta}) : \mathbb{R}^d \to \mathbb{R}^d$$

**The gradient captures the "shape" of the function** — for every $\vec{\theta}$ it tells us which way to go so that $f(\vec{\theta})$ **increases** fastest. So: start somewhere, take a small step in the **opposite** direction of the gradient, repeat.

**(Batch) Gradient Descent** — goal: minimize $R_N(\vec{\theta}) : \mathbb{R}^d \to \mathbb{R}$

```
k = 0,  θ⃗⁽⁰⁾ = θ⃗₀
while not converged:
    θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ − η_k ∇_θ R_N(θ⃗)|_{θ⃗ = θ⃗⁽ᵏ⁾}
    k = k + 1
```

**Stochastic Gradient Descent**

```
k = 0,  θ⃗⁽⁰⁾ = θ⃗₀
while not converged:
    randomly shuffle points
    for i = 1 … N:
        θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ − η_k ∇_θ loss(y⁽ⁱ⁾ θ⃗ · x⃗⁽ⁱ⁾)|_{θ⃗ = θ⃗⁽ᵏ⁾}
        k = k + 1
```

The difference is **where the $\frac{1}{N}\sum$ sits**: batch GD averages over the whole dataset before one step; SGD takes a step per example. Same objective, different granularity.

---

## 3. In-Class Exercise #5 — Convergence Criteria

> Which of the following are reasonable convergence criteria?
> **(a)** Check if training error is zero, same as in perceptron.
> **(b)** Check if empirical risk is zero.
> **(c)** Check if empirical risk stops changing.
> **(d)** Check if the parameter vector stops changing.

| | Reasonable? | Why |
|---|---|---|
| **(a)** $\mathcal{E}_N = 0$ | **No** | Only reachable if the data is **linearly separable** — the exact assumption we dropped in Lecture 05. On non-separable data the loop never exits. |
| **(b)** $R_N = 0$ | **No** | Worse than (a): hinge loss charges for correct-but-low-margin points, so $R_N = 0$ demands *every* point have margin $\ge 1$. Strictly stronger than separability. |
| **(c)** $\lvert \Delta R_N\rvert < \epsilon$ | **Yes** | Measures **change**, not an absolute target. Works whether or not the optimum is zero. |
| **(d)** $\lVert\vec{\theta}^{(k+1)} - \vec{\theta}^{(k)}\rVert < \epsilon$ | **Yes** | Same logic, in parameter space instead of objective space. |

**The pattern to remember:** with a relaxed objective, stop on **change**, not on **value**. (A max-iteration cap on top of either is standard practice.)

---

## 4. Recap: Update Rules for Hinge Loss

The two algorithms, written out for the hinge objective — worth seeing together because the batch form is the one usually skipped.

**SGD** (derived in Lecture 05 §8.1):

```
if y⁽ⁱ⁾ θ⃗⁽ᵏ⁾ · x⃗⁽ⁱ⁾ ≥ 1:
    no update
else:
    θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ + η_k y⁽ⁱ⁾ x⃗⁽ⁱ⁾
```

**Batch GD** — the same per-point term, averaged, with an indicator selecting the active ones:

$$\vec{\theta}^{(k+1)} = \vec{\theta}^{(k)} + \eta_k\,\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[y^{(i)}\,\vec{\theta}^{(k)}\cdot\vec{x}^{(i)} < 1\right]\left(y^{(i)}\vec{x}^{(i)}\right)$$

Read the indicator as "sum only over the margin violators." The `if / no update` in SGD *is* this indicator — one branch per example instead of a mask over all of them.

---

## 5. Recap: Learning Rate / Step Size

- **Too large** ⟹ **overshoot** — bounce across the minimum, possibly never converge.
- **Too small** ⟹ **slow** — many tiny steps to get anywhere.
- It is **a "hyperparameter" that can (and should) be tuned.**
- **Constant** vs. **decaying** learning rate; the slides' example schedule:

$$\eta_k = \frac{1}{1+k}$$

---

## 6. Recap: GD & SGD Convergence Theorems

> - With an appropriate learning rate, **if the objective function is convex, (S)GD will converge to the global minimum almost surely.**
> - **(S)GD is a general optimization algorithm** that can also be applied to **non-convex** objective functions — it converges to a **local** minimum.

That second bullet is why this machinery keeps showing up all semester: nothing in GD assumed linearity, convexity, or classification. It only assumed a gradient exists.

---

## 7. From Classification to Regression

$$\textbf{Classification task: } y \in \{-1,+1\} \ \text{ or } \ y \in \{0,1\} \qquad\qquad \textbf{Regression task: } y \in \mathbb{R}$$

The framing sentence is **identical** to the classification one — only the codomain moves:

> Given training data containing a set of labeled examples, come up with a rule that maps an unlabeled example to the correct label.

**Given** $\mathcal{D} = \{\vec{x}^{(i)}, y^{(i)}\}_{i=1}^{N}$, with $\vec{x}^{(i)} \in \mathbb{R}^d$ and $y \in \mathbb{R}$.
**Find** a "good" $f : \mathbb{R}^d \to \mathbb{R}$.

Note $f$, not $h$ — and no $\mathrm{sign}(\cdot)$ wrapping it. The model's raw output *is* the prediction now, which is exactly why a new loss is needed: there's no longer a notion of "correct side," only "how far off."

---

## 8. Linear Regression: The Setup

**Goal — learn a linear function of the feature vector:**

$$f(\vec{x}; \vec{\theta}, b) = \vec{\theta}\cdot\vec{x} + b$$

The two questions the slides pose, which structure the rest of the lecture:

1. **How do we measure "error"?** What criterion should we use to judge if a regression model is "good"?
2. **What algorithms can we use to optimize this criterion?**

*(As in Lecture 05, the offset $b$ is folded in by augmentation — $\vec{x}' = [1,\vec{x}]^T$, $\vec{\theta}' = [b,\vec{\theta}]^T$ — and dropped from the notation from here on.)*

---

## 9. Squared Loss

Empirical risk is still just a **proxy for training error**, and still has the same shape with a swappable interior:

$$R_N(\vec{\theta}) = \frac{1}{N}\sum_{i=1}^{N}\mathrm{loss}\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)$$

**Note the argument changed.** In classification the loss took the **margin** $z = y^{(i)}(\vec{\theta}\cdot\vec{x}^{(i)})$ — a *product*. In regression it takes the **residual** $z = y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}$ — a *difference*. Same letter $z$, completely different quantity; this is the single easiest thing to mix up in this lecture.

$$\boxed{\mathrm{loss}(z) = \frac{z^2}{2}, \qquad z = y - \vec{\theta}\cdot\vec{x}}$$

Also known as **"ordinary least squares" (OLS)**. The professor's margin note on the geometry: **errors are measured parallel to the $y$-axis** — *not* perpendicular to the fitted line. (Perpendicular distance is a different method entirely; that's total least squares / PCA territory.)

**Properties** — the same checklist as hinge loss, and squared loss passes it more cleanly:

- **continuous**
- **differentiable** — everywhere, with no kink (hinge loss has one at $z=1$; see Lecture 05 Appendix)
- **convex**

**Intuition for the shape:** a parabola centered at $z=0$ **permits small discrepancies but penalizes large deviations.** Near zero the penalty is negligible ($z=0.1 \Rightarrow 0.005$); far out it grows quadratically ($z=10 \Rightarrow 50$). That's a modeling choice, not a law — it says large errors are disproportionately bad, which also makes OLS **sensitive to outliers**.

> **The loss functions so far**, collected (the professor's boxed margin list):
>
> | Loss | Lecture | Task |
> |---|---|---|
> | zero-one loss | Lec 3 | classification |
> | perceptron loss | Lec 4 | classification |
> | hinge loss | Lec 5 | classification |
> | **squared loss** | **Lec 6** | **regression** ($y$ continuous) |
> | logistic loss | Lec 9 | classification |
>
> For the classification losses, $z = y^{(i)}(\vec{\theta}\cdot\vec{x}^{(i)})$.

**The objective**, then — minimize empirical risk with squared loss:

$$\boxed{R_N(\vec{\theta}) = \frac{1}{N}\sum_{i=1}^{N}\frac{\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)^2}{2}}$$

The $\frac{1}{2}$ is a convenience: it cancels against the $2$ from differentiating the square. It changes the value of the objective but not its minimizer.

---

## 10. Solution 1: SGD for Least Squares

**What algorithm can we use to optimize $R_N(\vec{\theta})$?** One we've already seen — **SGD**.

### 10.1 Deriving the update rule

Start from the generic SGD step:

$$\vec{\theta}^{(k+1)} = \vec{\theta}^{(k)} - \eta\,\left.\nabla_{\vec{\theta}}\,\mathrm{loss}\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)\right|_{\vec{\theta}=\vec{\theta}^{(k)}}$$

and differentiate the single-example squared loss by the chain rule:

$$\nabla_{\vec{\theta}}\ \frac{\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)^2}{2} = \frac{1}{2}\cdot 2\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)\cdot\nabla_{\vec{\theta}}\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right) = \left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)\left(-\vec{x}^{(i)}\right)$$

using the one identity from Lecture 05, $\nabla_{\vec{\theta}}(\vec{\theta}\cdot\vec{x}) = \vec{x}$. The two minus signs cancel:

$$\boxed{\vec{\theta}^{(k+1)} = \vec{\theta}^{(k)} + \eta\left(y^{(i)} - \vec{\theta}^{(k)}\cdot\vec{x}^{(i)}\right)\vec{x}^{(i)}}$$

> **Calc refresher** (the professor's boxed aside):
> **Chain rule:** $\frac{d}{dx}f(g(x)) = f'(g(x))\,g'(x)$.
> **Example:** $\frac{d}{dx}\frac{(y-wx)^2}{2} = (y-wx)\cdot(-w)$, taking $f(z) = \frac{z^2}{2}$ with $f'(z) = z$, and $g(x) = y - wx$ with $g'(x) = -w$.

**Read the update rule.** The step is $\vec{x}^{(i)}$ scaled by the **signed residual**:

- predicted too **low** ($y^{(i)} > \vec{\theta}\cdot\vec{x}^{(i)}$) ⟹ move $\vec{\theta}$ **along** $\vec{x}^{(i)}$, raising the prediction;
- predicted too **high** ⟹ move **against** $\vec{x}^{(i)}$;
- the **size** of the step scales with **how wrong** we were.

### 10.2 The algorithm

```
θ⃗⁽⁰⁾ = 0⃗,  k = 0
while convergence criteria not met:
    randomly shuffle points
    for i = 1 … N:
        θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ + η_k (y⁽ⁱ⁾ − θ⃗⁽ᵏ⁾ · x⃗⁽ⁱ⁾) x⃗⁽ⁱ⁾
        k++
```

> **Compared to hinge loss in classification:**
>
> - **Convergence criteria and learning-rate considerations → same as before.** Nothing new to learn (see §3, §5).
> - **The difference:** here we make **an update for every example unless the error is exactly 0.**
>
> That's a real behavioral change. Hinge loss has a **flat region** ($z \ge 1$) where the gradient is genuinely zero, so confidently-correct points are skipped. Squared loss is a parabola — its gradient is zero at **exactly one point**, $z = 0$. Since a real-valued prediction is essentially never exactly right, **every point contributes an update, forever.** This is why regression needs a change-based convergence criterion rather than "no more mistakes."

---

## 11. Solution 2: Closed-Form Solution

> Since $R_N(\vec{\theta})$ with squared loss is a **convex** function and **differentiable everywhere**, we can *try* to minimize it directly by setting the gradient to zero and solving analytically.

The emphasis on "try" is the professor's — §12 is where it doesn't always work.

> **Calc refresher** (the boxed aside, the 1-D version of the whole argument):
> Let $f(x) = (ax+b)^2$. $f$ is convex, i.e. concave up. Find the $x^*$ that minimizes $f$ — the **"critical point"**:
> 1. **find the first derivative:** $f'(x) = 2a(ax+b)$
> 2. **set $f'(x)$ to 0 and solve for $x$:** $2a(ax+b) = 0 \Rightarrow x^* = -\frac{b}{a}$
>
> Convexity is what makes this legitimate — for a convex function the critical point is the **global** minimum, not just any stationary point.

### 11.1 First, rewrite empirical risk in matrix notation (see HW1)

$$X = \begin{bmatrix} -\ \vec{x}^{(1)T}\ - \\ \vdots \\ -\ \vec{x}^{(N)T}\ - \end{bmatrix} \ \ N\times d \text{ matrix}, \qquad \vec{y} = \begin{bmatrix} y^{(1)} \\ \vdots \\ y^{(N)} \end{bmatrix} \ \ N \times 1 \text{ column vector}$$

One example **per row** of $X$ — the same layout as a pandas dataframe, and the same one HW1 used.

$$
\begin{aligned}
R_N(\vec{\theta}) &= \frac{1}{N}\sum_{i=1}^{N}\frac{1}{2}\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)^2 \\
&= \frac{1}{N}\cdot\frac{1}{2}\left(X\vec{\theta} - \vec{y}\right)^T\left(X\vec{\theta} - \vec{y}\right) \qquad\text{or equivalently}\quad \frac{1}{N}\cdot\frac{1}{2}\left\lVert X\vec{\theta} - \vec{y}\right\rVert_2^2 \\
&= \frac{1}{N}\cdot\frac{1}{2}\left(\vec{\theta}^TX^TX\vec{\theta} - \vec{\theta}^TX^T\vec{y} - \vec{y}^TX\vec{\theta} + \vec{y}^T\vec{y}\right) \\
&= \frac{1}{N}\cdot\frac{1}{2}\left(\vec{\theta}^T(X^TX)\vec{\theta} - 2\vec{\theta}^T(X^T\vec{y}) + \vec{y}^T\vec{y}\right)
\end{aligned}
$$

The last step folds the two middle terms together: both $\vec{\theta}^TX^T\vec{y}$ and $\vec{y}^TX\vec{\theta}$ are **scalars**, and each is the transpose of the other, so they're **equal** — hence the factor of 2. Worth pausing on; it's the step that makes the gradient clean.

### 11.2 Step 1: find the gradient

Two gradient identities do all the work (full table in the Appendix):

$$\nabla_{\vec{\theta}}\left(\vec{\theta}^TA\vec{\theta}\right) = \left(A + A^T\right)\vec{\theta}, \qquad \nabla_{\vec{\theta}}\left(\vec{\theta}^T\vec{v}\right) = \nabla_{\vec{\theta}}\left(\vec{v}^T\vec{\theta}\right) = \vec{v}$$

Apply them with $A = X^TX$, which is **symmetric**, so $(A + A^T) = 2A$ and the $\frac12$ cancels:

$$\nabla_{\vec{\theta}}R_N(\vec{\theta}) = \frac{1}{N}\left(\left(X^TX\right)\vec{\theta} - \left(X^T\vec{y}\right)\right)$$

($\vec{y}^T\vec{y}$ has no $\vec{\theta}$ in it, so it drops out.)

### 11.3 Step 2: set the gradient to zero and solve

$$
\begin{aligned}
\left.\nabla_{\vec{\theta}}R_N(\vec{\theta})\right|_{\vec{\theta}=\vec{\theta}^*} &= \vec{0} \\
\left(X^TX\right)\vec{\theta}^* - X^T\vec{y} &= \vec{0} \\
\left(X^TX\right)\vec{\theta}^* &= X^T\vec{y} \qquad\qquad \text{← the "normal equations"} \\
\boxed{\vec{\theta}^* = \left(X^TX\right)^{-1}X^T\vec{y}}
\end{aligned}
$$

Note the $\frac{1}{N}$ vanished the moment we set the gradient to $\vec{0}$ — scaling an objective by a positive constant doesn't move its minimizer. **No learning rate, no iterations, no convergence criterion.** This is the payoff for a quadratic objective, and it's why linear regression is the one model this semester with an exact solution.

---

## 12. Practical Notes

### 12.1 The OLS solution and its cost

$$\vec{\theta}^* = \left(X^TX\right)^{-1}X^T\vec{y} \qquad \textbf{when } X^TX \textbf{ is invertible.}$$

**Computational complexity of inverting $X^TX$ is $O(d^3)$** — it's a $d \times d$ matrix. So the closed form is cheap in $N$ but **cubic in the number of features**. That's the practical trade-off against SGD: with $d$ large, iterate; with $d$ modest, solve.

### 12.2 Invertibility of $X^TX$ (the "Gram matrix")

**Why might it be non-invertible / singular / degenerate / rank-deficient?**

Recall (see HW1):

$$\mathrm{rank}\left(X^TX\right) \le \mathrm{rank}(X) \le \min(N, d)$$

- **When $d$ is large, there may be redundant features** such that the columns of $X$ are **not linearly independent** — **"multicollinearity."**
- **E.g. if $d > N$:** $X^TX$ is a $d\times d$ matrix with rank **at most $N$** ⟹ **not invertible** ⟹ $\vec{\theta}^*$ **is not unique.**
- **E.g. a duplicated feature** $x_1 = x_2$: if $[\theta_1, \theta_2]$ is a solution, then so is $[\theta_1 + c,\ \theta_2 - c]$ for any $c$. An entire line of equally-good solutions — the objective literally cannot tell them apart.

> Note this is a statement about the **problem**, not the algorithm. SGD doesn't crash on collinear features; it just wanders along that flat direction and lands wherever initialization and shuffling take it.

### 12.3 What to do if $X^TX$ is not invertible

**Option 1 — the Moore–Penrose pseudoinverse:**

$$\vec{\theta}^* = \left(X^TX\right)^{+}X^T\vec{y} = X^{+}\vec{y}$$

- **Use `scipy.linalg.pinv` in HW2.**
- $A^{+}$ **is equivalent to $A^{-1}$ if $A$ is invertible** — so it's a safe default, not a special case.
- Otherwise it gives the **"minimum-norm" solution**: among all the equally-optimal $\vec{\theta}$, the one with the smallest $\lVert\vec{\theta}\rVert$. For the duplicated feature $x_1 = x_2$, that means $\theta_1 = \theta_2$ — **the weight of the duplicated feature is split equally** rather than dumped arbitrarily on one copy.

**Option 2 — Regularization** (next lecture). Note where this lands: preferring the smaller-norm solution when the data can't distinguish them is *exactly* what Lecture 07 formalizes. Ridge regression makes $X^TX + \lambda I$ **always** invertible.

---

## Appendix A: Common Differentiation Rules

The reference table from the slides — the multivariate column is the one worth memorizing.

| **derivatives** | | | **gradients** | |
|---|---|---|---|---|
| $f(x)$ | $\frac{d}{dx}f(x)$ | | $f(\vec{x})$ | $\nabla_{\vec{x}}f(\vec{x})$ |
| $ax$ | $a$ | | $\vec{v}^T\vec{x}$ | $\vec{v}$ |
| $x^2$ | $2x$ | | $\vec{x}^T\vec{x}$ | $2\vec{x}$ |
| $ax^2$ | $2ax$ | | $\vec{x}^TA\vec{x}$ | $(A + A^T)\vec{x}$; $= 2A\vec{x}$ **if $A$ symmetric** |

Each gradient rule is the direct analogue of the derivative beside it. $X^TX$ being symmetric is what makes the $2A\vec{x}$ shortcut apply in §11.2.

## Appendix B: Closed-Form Solution in Alternative Notation

The same result derived from the **summation** form instead of the matrix form — useful because it shows *where* $X^TX$ and $X^T\vec{y}$ come from, rather than assuming them.

**Step 1: find the gradient.** Start from what we already derived for SGD with squared loss:

$$
\begin{aligned}
\nabla_{\vec{\theta}}R_N(\vec{\theta}) &= \frac{1}{N}\sum_{i=1}^{N}\left(y^{(i)} - \vec{\theta}\cdot\vec{x}^{(i)}\right)\left(-\vec{x}^{(i)}\right) \\
&= \frac{1}{N}\left(-\sum_{i=1}^{N}y^{(i)}\vec{x}^{(i)} + \sum_{i=1}^{N}\left(\vec{\theta}\cdot\vec{x}^{(i)}\right)\vec{x}^{(i)}\right) \\
&= \frac{1}{N}\left(-\sum_{i=1}^{N}y^{(i)}\vec{x}^{(i)} + \sum_{i=1}^{N}\vec{x}^{(i)}\vec{x}^{(i)T}\vec{\theta}\right) \\
&= \frac{1}{N}\left(-\underbrace{\left(\sum_{i=1}^{N}y^{(i)}\vec{x}^{(i)}\right)}_{d\times 1} + \underbrace{\left(\underbrace{\sum_{i=1}^{N}\vec{x}^{(i)}}_{d\times 1}\underbrace{\vec{x}^{(i)T}}_{1\times d}\right)}_{d\times d}\vec{\theta}\right)
\end{aligned}
$$

The third line is the key move — **rearranged so that we can factor out $\vec{\theta}$**:

$$\left(\vec{\theta}\cdot\vec{x}^{(i)}\right)\vec{x}^{(i)} = \vec{x}^{(i)}\left(\vec{x}^{(i)}\cdot\vec{\theta}\right) = \vec{x}^{(i)}\vec{x}^{(i)T}\vec{\theta}$$

which is legal because $\vec{\theta}\cdot\vec{x}^{(i)}$ is a **scalar** times a **vector** — reorderable — and then the scalar is rewritten as a $1\times d$ row times $\vec{\theta}$. Turning a dot product into an **outer product** times $\vec{\theta}$ is what makes $\vec{\theta}$ factorable.

**Step 2: set the gradient to zero.**

$$-\underbrace{\left(\sum_{i=1}^{N}y^{(i)}\vec{x}^{(i)}\right)}_{=\ X^T\vec{y}} + \underbrace{\left(\sum_{i=1}^{N}\vec{x}^{(i)}\vec{x}^{(i)T}\right)}_{=\ X^TX}\vec{\theta}^* = \vec{0} \quad\Longrightarrow\quad X^TX\vec{\theta}^* = X^T\vec{y} \quad\Longrightarrow\quad \vec{\theta}^* = \left(X^TX\right)^{-1}X^T\vec{y}$$

**Why those two identifications hold** (the boxed recall at the bottom of the slide) — with $X = [\vec{x}^{(1)}, \dots, \vec{x}^{(N)}]^T$ an $N\times d$ matrix and $\vec{y} = [y^{(1)},\dots,y^{(N)}]^T$ an $N\times 1$ column vector, $X^T$ is $d \times N$ with the examples as **columns**:

$$X^T\vec{y} = \vec{x}^{(1)}y^{(1)} + \dots + \vec{x}^{(N)}y^{(N)} = \sum_{i=1}^{N}y^{(i)}\vec{x}^{(i)} \qquad (d\times 1)$$

$$X^TX = \vec{x}^{(1)}\vec{x}^{(1)T} + \dots + \vec{x}^{(N)}\vec{x}^{(N)T} = \sum_{i=1}^{N}\vec{x}^{(i)}\vec{x}^{(i)T} \qquad (d\times d)$$

Both are just "matrix product = sum of outer products of corresponding column/row pairs." Same identity, read in two directions.

---

## Quick Reference

```
TASK TYPES
  classification   y ∈ {−1,+1} or {0,1}     find h: R^d → {−1,+1}
  REGRESSION       y ∈ R                    find f: R^d → R
  linear model     f(x⃗; θ⃗, b) = θ⃗ · x⃗ + b        (b folded in by augmentation)

SQUARED LOSS  (a.k.a. ordinary least squares, OLS)
  loss(z) = z²/2  with  z = y − θ⃗ · x⃗      ← RESIDUAL (a difference),
                                              NOT the margin y(θ⃗·x⃗) (a product)
  continuous, differentiable EVERYWHERE, convex        (no kink, unlike hinge)
  permits small discrepancies, penalizes large deviations  ⇒ outlier-sensitive
  errors measured PARALLEL TO THE y-AXIS (not perpendicular to the line)

  R_N(θ⃗) = (1/N) Σ (y⁽ⁱ⁾ − θ⃗ · x⃗⁽ⁱ⁾)² / 2

SOLUTION 1 — SGD
  ∇_θ [ (y⁽ⁱ⁾ − θ⃗·x⃗⁽ⁱ⁾)²/2 ] = (y⁽ⁱ⁾ − θ⃗·x⃗⁽ⁱ⁾)(−x⃗⁽ⁱ⁾)
  θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ + η_k (y⁽ⁱ⁾ − θ⃗⁽ᵏ⁾ · x⃗⁽ⁱ⁾) x⃗⁽ⁱ⁾        step ∝ SIGNED residual
  vs. hinge: convergence & learning rate identical, BUT an update happens for
  every example unless the error is exactly 0 (no flat region to sit in)

SOLUTION 2 — CLOSED FORM   (works because the objective is quadratic ⇒ convex)
  X: N×d (one example per ROW),  y⃗: N×1
  R_N(θ⃗) = (1/N)(1/2) ‖Xθ⃗ − y⃗‖²₂
         = (1/N)(1/2)( θ⃗ᵀ(XᵀX)θ⃗ − 2θ⃗ᵀ(Xᵀy⃗) + y⃗ᵀy⃗ )
  ∇_θ R_N = (1/N)( (XᵀX)θ⃗ − Xᵀy⃗ ) = 0⃗
  normal equations:  (XᵀX) θ⃗* = Xᵀy⃗
  ⇒  θ⃗* = (XᵀX)⁻¹ Xᵀy⃗            no η, no iterations, no convergence test

GRADIENT IDENTITIES
  ∇_θ (θ⃗ · x⃗)   = x⃗            ∇_θ (v⃗ᵀθ⃗) = v⃗
  ∇_θ (θ⃗ᵀθ⃗)     = 2θ⃗           ∇_θ (θ⃗ᵀAθ⃗) = (A + Aᵀ)θ⃗  = 2Aθ⃗ if A symmetric
  Xᵀy⃗ = Σ y⁽ⁱ⁾x⃗⁽ⁱ⁾  (d×1)        XᵀX = Σ x⃗⁽ⁱ⁾x⃗⁽ⁱ⁾ᵀ  (d×d)

PRACTICAL
  cost of inverting XᵀX ......... O(d³)   (d×d "Gram matrix")
  rank(XᵀX) ≤ rank(X) ≤ min(N,d)         (HW1)
  non-invertible when columns of X are dependent ("multicollinearity"),
    e.g. d > N ⇒ rank ≤ N ⇒ θ⃗* NOT UNIQUE
    e.g. x₁ = x₂ ⇒ [θ₁+c, θ₂−c] is also optimal for any c
  fix 1: pseudoinverse  θ⃗* = (XᵀX)⁺Xᵀy⃗ = X⁺y⃗   (scipy.linalg.pinv, HW2)
         A⁺ = A⁻¹ when invertible; else the MINIMUM-NORM solution
         (x₁ = x₂ ⇒ θ₁ = θ₂, duplicated weight split equally)
  fix 2: regularization → Lecture 07

CONVERGENCE CRITERIA (In-Class Ex #5):  stop on CHANGE, not on VALUE
  (a) E_N = 0   ✗ needs separability      (c) |ΔR_N| < ε              ✓
  (b) R_N = 0   ✗ even stronger           (d) ‖θ⃗⁽ᵏ⁺¹⁾ − θ⃗⁽ᵏ⁾‖ < ε    ✓
```
