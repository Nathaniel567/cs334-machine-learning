# CS-334: Machine Learning — Fall 2026

Emory University · Department of Computer Science · Section 2
Instructor: **Prof. Shengpu Tang** · Office hours Tue/Thu 2:15–3:15pm (after class), or by appointment

Coursework, lecture notes, and in-class exercises.

---

## Repository Layout

```
.
├── lectures/
│   ├── slides/          original lecture PDFs
│   └── notes/           my written notes, one markdown file per lecture
├── exercises/           in-class exercises, one directory each
│   ├── ex02-python-numpy-pandas/
│   └── ex03-linear-classification/
└── homework/
    └── hw1/             spec, skeleton code, data, self-check harness
```

---

## Lectures

| # | Date | Topic | Slides | Notes |
|---|---|---|---|---|
| 02 | 09/01 | Python Crash Course | [PDF](lectures/slides/Lecture-02-Python-Crash-Course.pdf) | [Notes](lectures/notes/Lecture-02-Python-Crash-Course.md) |
| 03 | 09/03 | Linear Classification | [PDF](lectures/slides/Lecture-03-Linear-Classification.pdf) | [Notes](lectures/notes/Lecture-03-Linear-Classification.md) |
| 04 | 09/08 | Perceptron | [PDF](lectures/slides/Lecture-04-Perceptron.pdf) | [Notes](lectures/notes/Lecture-04-Perceptron.md) |
| 05 | 09/10 | Gradient Descent | [PDF](lectures/slides/Lecture-05-Gradient-Descent.pdf) | [Notes](lectures/notes/Lecture-05-Gradient-Descent.md) |
| 06 | 09/15 | Linear Regression | [PDF](lectures/slides/Lecture-06-Linear-Regression.pdf) | [Notes](lectures/notes/Lecture-06-Linear-Regression.md) |
| 07 | 09/17 | Regularization | [PDF](lectures/slides/Lecture-07-Regularization.pdf) | [Notes](lectures/notes/Lecture-07-Regularization.md) |

## In-Class Exercises

| # | Topic | Status | Files |
|---|---|---|---|
| 02 | NumPy / pandas / matplotlib on the iris dataset | Submitted | [answers](exercises/ex02-python-numpy-pandas/Exercise-02-Answers.md) · [`solve.py`](exercises/ex02-python-numpy-pandas/solve.py) · [figures](exercises/ex02-python-numpy-pandas/figures) |
| 03 | ML terminology, linear decision boundaries | Submitted · answers verified against the Lecture 04 solutions | [answers](exercises/ex03-linear-classification/Exercise-03-Answers.md) |

## Homework

| # | Due | Status | Files |
|---|---|---|---|
| 1 | Sun 09/13, 11:59pm | Submitted | [`homework/hw1/`](homework/hw1) |
| 2 | Sun 09/27, 11:59pm | Released — not started | — (extra credit: Weighted Linear Regression, due Wed 09/30) |

---

## Course Reference

**The running notation.** Every lecture so far builds on this, so it's worth having in one place:

| Symbol | Meaning |
|---|---|
| $\vec{x} = [x_1,\dots,x_d]^T \in \mathbb{R}^d$ | feature vector |
| $y \in \{-1,+1\}$ | binary label |
| $\mathcal{D} = \{(\vec{x}^{(i)}, y^{(i)})\}_{i=1}^{N}$ | training set of $N$ labeled examples |
| $h: \mathbb{R}^d \to \{-1,+1\}$ | classifier |
| $\mathcal{H}$ | hypothesis class — the set of classifiers we search over |
| $\vec{\theta} \in \mathbb{R}^d$ | parameter / weight / normal vector — *is* the model |
| $b \in \mathbb{R}$ | offset (intercept) |
| $z = y(\vec{\theta}\cdot\vec{x})$ | margin — sign says right/wrong, magnitude says how much |
| $R_N(\vec{\theta})$ | empirical risk — training error with the 0-1 indicator relaxed to a loss |
| $\eta_k$ | step size / learning rate at step $k$ |
| $f: \mathbb{R}^d \to \mathbb{R}$ | regressor (Lec 06 on — $y \in \mathbb{R}$, no $\mathrm{sign}(\cdot)$) |
| $X \in \mathbb{R}^{N\times d}$, $\vec{y} \in \mathbb{R}^{N}$ | design matrix (one example per **row**) and label vector |
| $\phi(x)$ | explicit feature mapping, e.g. $[1,x,x^2,\dots,x^M]$ |
| $\Omega(\vec{\theta})$, $\lambda > 0$ | regularizer and regularization strength |

```
h(x⃗; θ⃗)      = sign(θ⃗ · x⃗)                          linear classifier through origin
h(x⃗; θ⃗, b)   = sign(θ⃗ · x⃗ + b)                      with offset
E_N(θ⃗)       = (1/N) Σ 1[ y⁽ⁱ⁾(θ⃗ · x⃗⁽ⁱ⁾) ≤ 0 ]      training error
R_N(θ⃗)       = (1/N) Σ loss(y⁽ⁱ⁾, θ⃗ · x⃗⁽ⁱ⁾)        empirical risk
loss_h(z)    = max{0, 1 − z}                       hinge loss (convex)
θ⃗ ← θ⃗ − η ∇f(θ⃗)                                   gradient descent step

f(x⃗; θ⃗, b)   = θ⃗ · x⃗ + b                            linear regression
loss(z)      = z²/2,  z = y − θ⃗ · x⃗                 squared loss / OLS (RESIDUAL,
                                                     not the margin y(θ⃗·x⃗))
θ⃗* = (XᵀX)⁻¹ Xᵀy⃗                                    OLS closed form
J(θ⃗)         = R_N(θ⃗) + λ Ω(θ⃗)                     regularized objective
θ⃗* = (XᵀX + λI)⁻¹ Xᵀy⃗                               ridge (always invertible)
```

Three things that are easy to lose points on:

- **Points exactly on the decision boundary count as misclassified** — the training-error indicator uses $\leq 0$, not $< 0$.
- **The length of $\vec{\theta}$ never matters, only its direction.** Scaling by any $c>0$ gives an identical classifier — so infinitely many $\vec{\theta}$ are equally correct.
- **A classifier through the origin can never classify a point at the origin**, since $\vec{\theta}\cdot\vec{0}=0$ for every $\vec{\theta}$. This is a large part of why the offset $b$ exists.

## Logistics

- **Gradescope:** course `1341911`, entry code `X8BBB7`. Homework submits in two parts (`HW#-Written` PDF and `HW#-Code`); the autograder accepts multiple submissions.
- **Canvas:** slides, linear algebra + multivariable calculus refresher notes, in-class exercise data files, sample solution code.
- **Demo repo:** <https://github.com/shengpu-tang/CS334-demo> (`Python_Tutorial.ipynb` → "Open in Colab")

## Environment

Python 3.12 with `numpy`, `pandas`, `matplotlib`. Colab has these preinstalled; locally:

```bash
pip3 install numpy pandas matplotlib
```
