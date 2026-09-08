# CS-334 Machine Learning — Lecture 03: Linear Classification
**Date:** 09/03/2026 · **Instructor:** Prof. Shengpu Tang
**Slides:** `lectures/slides/Lecture-03-Linear-Classification.pdf`
**Paired exercise:** In-Class Exercise #3 → `exercises/ex03-linear-classification/`

---

## Admin / Logistics

- Linear algebra + multivariable calculus refresher notes are on Canvas.
- **HW1** out, due **Sun 09/13, 11:59pm** on Gradescope. The autograder accepts multiple submissions.
- Office hours: Prof. Tang, Tue/Thu after class **2:15–3:15pm**, or by appointment. Other OH on Canvas.

---

## 1. Setting Up the Supervised Classification Problem

Four objects define the problem. Get comfortable with this notation — the whole course reuses it.

| Object | Notation | Meaning |
|---|---|---|
| Feature vector | $\vec{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$ | One example, described by $d$ numbers |
| Label | $y \in \{-1, +1\}$ | Binary class. Note: **$\pm 1$**, not $\{0,1\}$ |
| Training set | $\mathcal{D} = \{(\vec{x}^{(i)}, y^{(i)})\}_{i=1}^{N}$ | $N$ labeled examples |
| Classifier | $h: \mathbb{R}^d \to \{-1, +1\}$ | A function mapping features to a label |

> **Why $\pm1$ instead of $\{0,1\}$?** Because it makes the "is this prediction correct?" test a single product:
> $y \cdot h(\vec{x}) > 0$ ⟺ correct. This shows up everywhere below and in the perceptron update rule.

**Goal.** Select the best $h$ from a set of possible classifiers $\mathcal{H}$ (the *hypothesis class*) — best meaning *the best chance of correctly classifying new examples*, not the ones we already have.

**How.** Via a **learning algorithm**, which is typically an **optimization problem with respect to $\mathcal{D}$**.

---

## 2. The Motivating Failure: Why You Can't Just Memorize

**Example — ECG data sampled at 360 Hz.** One second of signal gives

$$\vec{x} = [x_1, x_2, \dots, x_{360}]^T, \qquad x_k \in \{1, \dots, 2048\}$$

Suppose $N = 100$, $\mathcal{D} = \{(\vec{x}^{(i)}, y^{(i)})\}_{i=1}^{100}$, and coordinate $x_6$ happens to be **different in every example**.

**Trivial "solution" — a lookup table:**

$$h(\vec{x}) = \begin{cases} y^{(i)} & \text{if } x_6 = x_6^{(i)} \\ ? & \text{otherwise} \end{cases}$$

This gets **zero training error**. Is it a good classifier? **No — it overfits.** It has memorized an accident of the training set and says nothing at all about a new patient (the `?` branch is undefined).

- **Generalization** = works well on *unseen* examples. That, not training error, is the actual objective.
- **Problem:** too many choices in $\mathcal{H}$. If $\mathcal{H}$ contains every possible function, one of them is always this lookup table.
- **Solution (?):** **constrain $\mathcal{H}$**. But it can't be *too* small, or we end up unable to classify even $\mathcal{D}$ itself.
- Finding the right balance is **model selection** — a recurring theme for the rest of the course.

Restricting $\mathcal{H}$ to **linear classifiers** is the first constraint we impose.

---

## 3. Linear Classifier Through the Origin

A **thresholded linear mapping** from feature vectors to labels:

$$h(\vec{x}; \vec{\theta}) = \begin{cases} +1 & \text{if } \vec{\theta} \cdot \vec{x} > 0 \\ -1 & \text{if } \vec{\theta} \cdot \vec{x} < 0 \end{cases}
\qquad \vec{\theta} = [\theta_1, \theta_2, \dots, \theta_d]^T \in \mathbb{R}^d$$

The inner quantity is just a **linear combination of the input features**:

$$\vec{\theta} \cdot \vec{x} = \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_d x_d$$

Different $\vec{\theta}$'s produce (potentially) different labelings of the same $\vec{x}$. **$\vec{\theta}$ is the entire model** — learning = picking $\vec{\theta}$.

### 3.1 Why the dot product decides the label

Use the geometric form of the dot product:

$$\vec{\theta} \cdot \vec{x} = \|\vec{\theta}\|\,\|\vec{x}\|\cos\alpha, \qquad \|\vec{x}\| = \sqrt{x_1^2 + x_2^2 + \dots + x_d^2}$$

Both norms are **always $\geq 0$**. So the sign of $\vec{\theta}\cdot\vec{x}$ is **entirely determined by $\cos\alpha$**, where $\alpha$ is the angle between $\vec{\theta}$ and $\vec{x}$:

| Angle $\alpha$ | $\cos\alpha$ | Predicted label |
|---|---|---|
| $0° \le \alpha < 90°$ | $> 0$ | $+1$ |
| $\alpha = 90°$ | $0$ | on the boundary |
| $90° < \alpha \le 180°$ | $< 0$ | $-1$ |

**⇒ The angle $\alpha$ determines the sign of $\vec{\theta} \cdot \vec{x}$.** Geometrically: is $\vec{x}$ on the same side as $\vec{\theta}$ points, or the opposite side?

### 3.2 The decision boundary

What if a point lies exactly on the boundary? Then $\alpha = 90°$:

$$\vec{\theta} \cdot \vec{x} = \|\vec{\theta}\|\,\|\vec{x}\|\cos 90° = 0$$

∴ **the hyperplane is defined by $\vec{\theta}\cdot\vec{x} = 0$.** In $d$ dimensions this hyperplane always **passes through the origin** ($\vec{x}=\vec{0}$ always satisfies it), and $\vec{\theta}$ is its **normal vector**.

In 2D, $\theta_1 x_1 + \theta_2 x_2 = 0$, and rearranging:

$$x_2 = -\frac{\theta_1}{\theta_2}\, x_1$$

so $-\theta_1/\theta_2$ is the **slope of the decision boundary in 2D**.

### 3.3 Two questions worth internalizing

**Does the *length* of $\vec{\theta}$ matter?** **No.** Scaling $\vec{\theta} \to c\vec{\theta}$ for any $c > 0$ leaves $\text{sign}(\vec{\theta}\cdot\vec{x})$ unchanged, so the classifier is identical. There are infinitely many $\vec{\theta}$ giving the same classifier.

**Does the *direction* matter?** **Yes** — and in two ways. It sets the orientation of the hyperplane, *and* flipping $\vec{\theta} \to -\vec{\theta}$ keeps the same boundary but **swaps which side is positive**.

> Informally: **$\vec{\theta}$ determines both the "orientation" and the "side" of the decision boundary.**

*(This scale-invariance is exactly the argument used in Exercise 3, Q3.2.)*

---

## 4. Choosing $\vec{\theta}$: Minimize Training Error

**Intuition:** find a $\vec{\theta}$ that works well on the training data $\mathcal{D}$.

> *"Wait, I thought you said overfitting? Why is it OK now?"*
> Because we've already **restricted the class of possible classifiers to linear ones**, which reduces the chance of overfitting. Fitting the training data is only dangerous when $\mathcal{H}$ is unconstrained.

**Training error** = the fraction of training examples for which the classifier predicts the wrong label:

$$\mathcal{E}_N(\vec{\theta}) = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[y^{(i)} \neq h(\vec{x}^{(i)}; \vec{\theta})\right]
= \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[y^{(i)}\left(\vec{\theta}\cdot\vec{x}^{(i)}\right) \leq 0\right]$$

- $\mathbb{1}[\cdot]$ is the **indicator function**: returns 1 if the logical expression is true, 0 otherwise.
- The two forms are equivalent because $y^{(i)}(\vec{\theta}\cdot\vec{x}^{(i)}) < 0$ exactly when $y^{(i)}$ and $\vec{\theta}\cdot\vec{x}^{(i)}$ **have opposite signs** — i.e. the prediction is wrong. *(This is the payoff of the $\pm1$ label encoding.)*
- **⚠️ Note the $\leq$, not $<$.** **By convention, points on the decision boundary are counted as misclassified.** Easy point to drop on a homework or exam.

### 4.1 Worked intuition (from the slides)

Same 8 points (four `+` upper right, four `−` lower left), three different $\vec{\theta}$:

| Boundary | Training error |
|---|---|
| Separates the two clusters cleanly, $\vec{\theta}$ pointing to the positive side | $0$ |
| Correct orientation but $\vec{\theta}$ **flipped** so the positive side faces the negatives | $1$ (every point wrong) |
| Slightly rotated so one point falls on the wrong side | $\tfrac{1}{9}$ |

Takeaway: orientation and sign of $\vec{\theta}$ are both doing real work, and training error is a **discrete count** — it jumps in steps as $\vec{\theta}$ rotates.

---

## 5. Where This Goes Next

Instead of trying all possible $\vec{\theta}$'s one by one (there are infinitely many, and $\mathcal{E}_N$ is a step function so calculus doesn't directly help), **how can we efficiently *learn* the best $\vec{\theta}$?**

→ **Lecture 04: the Perceptron algorithm.** See `Lecture-04-Perceptron.md`.

---

## Quick Reference

```
x⃗ ∈ ℝᵈ                          feature vector
y  ∈ {−1, +1}                    label
D  = {(x⃗⁽ⁱ⁾, y⁽ⁱ⁾)}ᵢ₌₁ᴺ          training set
θ⃗  ∈ ℝᵈ                          parameter / weight / normal vector

h(x⃗; θ⃗)   = sign(θ⃗ · x⃗)                        classifier
θ⃗ · x⃗ = 0                                       decision boundary (through origin)
θ⃗ · x⃗ = ‖θ⃗‖‖x⃗‖cos α                            sign set by angle α alone
x₂ = −(θ₁/θ₂)x₁                                 boundary slope in 2D
E_N(θ⃗)  = (1/N) Σ 1[ y⁽ⁱ⁾(θ⃗ · x⃗⁽ⁱ⁾) ≤ 0 ]      training error  (≤ : boundary = wrong)
```
