# CS-334 Machine Learning — Lecture 05: Gradient Descent
**Date:** 09/10/2026 · **Instructor:** Prof. Shengpu Tang
**Slides:** `lectures/slides/Lecture-05-Gradient-Descent.pdf` (includes In-Class Exercise #4)
**Prereq:** `Lecture-04-Perceptron.md`

---

## Admin / Logistics

- **HW1** due **Sun 11:59pm** on Gradescope. Autograder accepts multiple submissions.
- **Tag your PDF pages to question parts on Gradescope.** You can do it before *or* after the deadline — it does **not** change your submission timestamp. If a grader can't locate an answer because pages aren't assigned, they reserve the right to **not grade it or deduct points**.

---

## 0. Where This Lecture Sits

Lecture 04 ended on a cliff: the perceptron only works when the data is **linearly separable**. Today answers "what do we do when it isn't?"

The goal is unchanged — **generalize to unseen data**. What changes is the *objective function we optimize*, and the *algorithm we optimize it with*.

```
Lec 03:  objective = training error E_N        (NP-hard to minimize directly)
Lec 04:  algorithm = perceptron                (works only if separable)
Lec 05:  objective = empirical risk R_N        (a "relaxed" E_N)
         algorithm = (stochastic) gradient descent
```

---

## 1. Recap: Perceptron with Offset via Augmentation

Quick review of where Lecture 04 landed, because the same trick reappears.

$$\vec{x}' = [1, \vec{x}]^T, \qquad \vec{\theta}' = [b, \vec{\theta}]^T \quad\Longrightarrow\quad \vec{\theta}'\cdot\vec{x}' = b + \vec{\theta}\cdot\vec{x}$$

$$[b^{(k+1)}, \vec{\theta}^{(k+1)}]^T = [b^{(k)}, \vec{\theta}^{(k)}]^T + y^{(i)}[1, \vec{x}^{(i)}]^T$$

The offset update $b \mathrel{+}= y^{(i)}$ isn't a separate rule — it's the first coordinate of the ordinary update on the augmented vectors.

```
k = 0,  θ⃗⁽⁰⁾ = 0⃗,  b⁽⁰⁾ = 0
while not all points are correctly classified:
    for i = 1 … N:
        if y⁽ⁱ⁾ (θ⃗⁽ᵏ⁾ · x⃗⁽ⁱ⁾ + b⁽ᵏ⁾) ≤ 0:      ← a mistake-driven algorithm
            θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ + y⁽ⁱ⁾ x⃗⁽ⁱ⁾
            b⁽ᵏ⁺¹⁾  = b⁽ᵏ⁾ + y⁽ⁱ⁾
            k = k + 1
```

### 1.1 Properties, restated

> **Convergence Theorem.** If the training dataset is linearly separable, the perceptron is guaranteed to converge after a **finite** number of updates.

**However:**

- **No guarantee of uniqueness.** The solution depends on initialization and on the order of the data points — and some data points might never be used at all.
- **"Finite" can still be large.** Two reasons it drags:
  - **undershoot** — the current point remains misclassified even after its update (margin improves by only $\|\vec{x}^{(i)}\|^2$; see Lecture 04 §7);
  - a **previously correct point may become misclassified** by an update made for some other point.

**Worked trace from the slides** (2-D, through the origin): $\vec{\theta}^{(0)} = [0,0]$ → first mistake gives $\vec{\theta}^{(1)} = [1,2]$ → two points check out with no update → one more mistake gives $\vec{\theta}^{(2)} = [-2,2]$, the final boundary. Note the middle steps where nothing happens: **no mistake, no update.**

---

## 2. Historical Aside

Worth knowing because the hype cycle repeats.

- Rosenblatt's perceptron, 1958 — *Research Trends*: "Introducing the perceptron — a machine which senses, recognizes, remembers, and responds like the human mind."
- *The New York Times*, 1958: "…the embryo of an electronic computer that [was expected] to walk, talk, see, write, reproduce itself, and be conscious of its existence…"
- The **Mark 1 Perceptron** was physical hardware — a 400-pixel camera wired to motor-driven potentiometers as weights.
- **Minsky and Papert, *Perceptrons*, 1969** popped the bubble by pointing at **XOR**.

---

## 3. The XOR Problem: What Breaks

The **"famous XOR problem"**: four points, positives on one diagonal, negatives on the other. No line separates them — not through the origin, not with an offset.

On non-separable data:

- the perceptron **runs indefinitely and never converges** (the `while not all points correctly classified` condition can never be satisfied), **and**
- it is **not guaranteed to find a solution that minimizes training error** — it doesn't even give you a decent answer for the trouble.

> **The real lesson.** The problem isn't the algorithm; it's the **objective**. $\mathcal{E}_N$ is a sum of step functions, and there's no sane way to descend a staircase. Fix the objective and a general-purpose optimizer becomes available.

---

## 4. Why Training Error Is a Bad Objective Function

$$\mathcal{E}_N(\vec{\theta}) = \frac{1}{N}\sum_{i=1}^{N}\underbrace{\mathbb{1}\left[y^{(i)}\left(\vec{\theta}\cdot\vec{x}^{(i)}\right) \leq 0\right]}_{\text{indicator function}}$$

*(From here on the offset is **implicit** — assume $\vec{x}$ and $\vec{\theta}$ are the augmented vectors from §1. This is purely for conciseness.)*

### 4.1 The margin $z$, and what it tells us

Define the **margin** of an example:

$$z = y\left(\vec{\theta}\cdot\vec{x}\right)$$

This single scalar carries two pieces of information:

| | Meaning |
|---|---|
| **Sign of $z$** | whether the prediction is **right or wrong** ($z > 0$ right, $z \le 0$ wrong) |
| **Magnitude of $z$** | **how** right or wrong — because $\dfrac{\vec{\theta}\cdot\vec{x}}{\|\vec{\theta}\|}$ is the distance from the decision boundary, so $|z|$ is proportional to that distance |

Note $|y| = 1$, so $y^2 = 1$ and $y$ only ever flips the sign — all the magnitude lives in $\vec{\theta}\cdot\vec{x}$.

**This is the key reframing.** The 0-1 indicator throws the magnitude away and keeps only the sign. Everything that follows is about keeping the magnitude.

### 4.2 Empirical risk

Replace the indicator with a general **loss function** of the margin:

$$R_N(\vec{\theta}) = \frac{1}{N}\sum_{i=1}^{N}\mathrm{loss}\left(y^{(i)}, \vec{\theta}\cdot\vec{x}^{(i)}\right)$$

**Empirical risk is a "relaxed" version of training error** — same shape, swappable interior. Choosing

$$\mathrm{loss}_{0\text{-}1}(z) = \mathbb{1}[z \leq 0]$$

recovers $\mathcal{E}_N$ exactly: **empirical risk with zero-one loss *is* training error.**

### 4.3 What's wrong with zero-one loss

Plot $\mathrm{loss}_{0\text{-}1}(z)$ against $z$: it's $1$ for $z \le 0$, drops to $0$ for $z > 0$.

- **Not continuous** — a jump discontinuity at $z = 0$.
- **Not differentiable** at $z = 0$ (and the derivative is $0$ everywhere else, which is just as useless — no direction to move in).
- **Not convex.**

**Consequence:** direct minimization of empirical risk with 0-1 loss is **challenging in the general case — NP-hard**. The **loss surface** over $\vec{\theta}$ is a flat staircase: plateaus separated by cliffs, nothing to roll downhill on.

> **Calc refresher** (the professor wrote this out in the margin — it's the vocabulary the rest of the lecture leans on):
>
> **Continuous at $c$:** $\lim_{x\to c} f(x) = f(c)$, which requires all three of — the two one-sided limits exist, they're equal, and $f$ is defined at $c$.
>
> **Differentiable at $c$:** $f'(c) = \lim_{h\to 0}\frac{f(c+h)-f(c)}{h}$ exists, i.e. the one-sided limits of the difference quotient exist **and are equal**.
>
> **A differentiable function must also be continuous** (the converse is false — $|x|$ at $0$).

---

## 5. Hinge Loss

$$\boxed{\mathrm{loss}_h(z) = \max\{0,\ 1 - z\}}, \qquad z = y\left(\vec{\theta}\cdot\vec{x}\right)$$

A hinge: sloped line $1-z$ for $z < 1$, flat $0$ for $z \ge 1$.

**(Nicer) properties:**

- **continuous**
- **differentiable** (everywhere except the kink at $z=1$ — see §9)
- **convex**

**There exist simple algorithms that can optimize this.** That's the whole point of the swap.

Two things to notice about the shape:

- It **penalizes correct-but-barely** predictions. At $z = 0.5$ the prediction is right, yet the loss is $0.5$ — it wants margin, not just correctness. Zero-one loss would charge $0$ here.
- It's an **upper bound** on zero-one loss, so driving $R_N$ down drags $\mathcal{E}_N$ down with it.

The **loss surface** for hinge loss is a convex, piecewise-linear bowl — a surface gradient descent can actually walk down.

---

## 6. Gradient Descent

### 6.1 The intuition: Link's Foggy Descent

The professor's framing (Breath of the Wild):

> You're standing at the summit of Mount Lanayru. Suddenly a thick fog rolls in. You can't see the shrines, the towers, or even the path in front of you. You don't have a compass, and your Sheikah Slate is broken. **You can only feel the ground around your feet.** How do you get to the campfire at the bottom of the valley?

The answer:

- **Look for the direction of steepest descent** (all you can sense is local slope),
- **take a small step**,
- **repeat**.

That's the entire algorithm. The fog is the point: you never get to see the whole function, only its local shape.

### 6.2 The gradient

Let $f(\vec{\theta}): \mathbb{R}^d \to \mathbb{R}$ be some function. We want to find a $\vec{\theta}^*$ that minimizes $f(\vec{\theta})$.

$$\nabla_{\vec{\theta}}\, f(\vec{\theta}) : \mathbb{R}^d \to \mathbb{R}^d$$

**The gradient captures the "local shape" of the function.** For every $\vec{\theta}$, it tells us which way to go such that $f(\vec{\theta})$ **increases** the fastest.

Note the types: $f$ maps a vector to a **scalar**, but $\nabla f$ maps a vector to a **vector** of the same dimension.

### 6.3 The algorithm

Gradient points uphill ⟹ **step the opposite way**.

$$\boxed{\vec{\theta}^{(k+1)} = \vec{\theta}^{(k)} - \eta_k \left.\nabla_{\vec{\theta}} f(\vec{\theta})\right|_{\vec{\theta} = \vec{\theta}^{(k)}}}$$

```
k = 0,  θ⃗⁽⁰⁾ = θ⃗₀
while not converged:                          ← stopping criterion (§9)
    θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ − η_k ∇_θ f(θ⃗)|_{θ⃗ = θ⃗⁽ᵏ⁾}    ← η_k = step size / learning rate
    k = k + 1
```

Two things that are easy to get wrong: the sign is **minus** (we're descending), and the gradient is **evaluated at the current $\vec{\theta}^{(k)}$**, not at some fixed point.

### 6.4 Warm-up: one-dimensional gradient descent

Let $f(\theta) = \theta^2 - 10\theta$. Find $\theta^* = \arg\min_\theta f(\theta)$.

**In Calc 1:** set the derivative to zero. $f'(\theta) = 2\theta - 10 = 0 \Rightarrow \boxed{\theta^* = 5}$. Analytically solvable — that's why it's a warm-up.

**With gradient descent:** the update rule is

$$\theta^{(k+1)} = \theta^{(k)} - \eta\, f'(\theta^{(k)}) = \theta^{(k)} - \eta\left(2\theta^{(k)} - 10\right)$$

```
θ⁽⁰⁾ = 0
while not converged:
    θ⁽ᵏ⁺¹⁾ = θ⁽ᵏ⁾ − η (2θ⁽ᵏ⁾ − 10)
    k = k + 1
```

Starting at $\theta^{(0)} = 0$ with $\eta = 0.1$: $0 \to 1 \to 1.8 \to 2.44 \to \dots \to 5$. Each step covers $20\%$ of the remaining distance, so it approaches $5$ but never lands exactly — hence the need for a **convergence criterion** rather than an equality test.

The two questions this raises, both answered in §9: **what's the effect of the learning rate**, and **how do we decide convergence?**

### 6.5 Scaling up to multiple dimensions

| | Univariate | Multivariate |
|---|---|---|
| Function | $f(\theta),\ \theta \in \mathbb{R}$, $f:\mathbb{R}\to\mathbb{R}$ | $f(\vec{\theta}),\ \vec{\theta} \in \mathbb{R}^d$, $f:\mathbb{R}^d\to\mathbb{R}$ |
| Need | the **derivative** $f'(\theta)$ | the **gradient** $\nabla_{\vec\theta} f(\vec\theta)$ |
| Example | $f = \theta^2 - 10\theta \Rightarrow f' = 2\theta - 10$ | $\nabla f(\vec{\theta}) = \left[\frac{\partial f}{\partial \theta_1}, \dots, \frac{\partial f}{\partial \theta_d}\right]^T$ |
| Output type | **scalar-valued** function | **vector-valued** function |

**The one identity to memorize**, since it's the workhorse for every linear model this semester:

$$\nabla_{\vec{\theta}}\left(\vec{\theta}\cdot\vec{x}\right) = \nabla_{\vec{\theta}}\left(\theta_1 x_1 + \dots + \theta_d x_d\right) = [x_1, \dots, x_d]^T = \vec{x}$$

The gradient of a dot product w.r.t. $\vec\theta$ is just $\vec{x}$ — the linear-algebra analogue of $\frac{d}{d\theta}(a\theta) = a$.

---

## 7. Back to the Problem at Hand

We want to minimize empirical risk $R_N(\vec{\theta})$ w.r.t. $\vec{\theta}$ using gradient descent, so we need its gradient:

$$\nabla_{\vec{\theta}} R_N(\vec{\theta}) = \frac{1}{N}\sum_{i=1}^{N} \nabla_{\vec{\theta}}\, \mathrm{loss}\left(y^{(i)}, \vec{\theta}\cdot\vec{x}^{(i)}\right)$$

> **The problem:** that summation means we have to look at **every data point before making a single update**. That's **slow** — one step of progress costs a full pass over the dataset, and datasets are large.

**Instead, we can update $\vec{\theta}$ based on a single point.** That's the whole idea of the stochastic variant.

---

## 8. Stochastic Gradient Descent (SGD)

```
k = 0,  θ⃗⁽⁰⁾ = θ⃗₀
while convergence criterion not met:
    randomly shuffle data points              ← the "stochastic" part
    for i = 1 … N:
        θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ − η_k ∇_θ loss(y⁽ⁱ⁾, θ⃗ · x⃗⁽ⁱ⁾)
        k = k + 1
```

The $\frac{1}{N}\sum$ is gone: each **individual** point produces an update. Each step is a noisy estimate of the true gradient, but you take $N$ times as many of them per pass.

### 8.1 In-Class Exercise #4 — derive the SGD update rule for hinge loss

$$\mathrm{loss}_h\left(y, \vec{\theta}\cdot\vec{x}\right) = \max\left\{0,\ 1 - y\left(\vec{\theta}\cdot\vec{x}\right)\right\}$$

Split on the two branches of the $\max$:

**Case 1: $y(\vec{\theta}\cdot\vec{x}) \geq 1$.** The loss is **zero**, which is its minimum, so the gradient is also $\vec{0}$ — **no update**.

**Case 2: $y(\vec{\theta}\cdot\vec{x}) < 1$.** The active branch is $1 - y(\vec{\theta}\cdot\vec{x})$:

$$\nabla_{\vec{\theta}}\,\mathrm{loss}_h = \nabla_{\vec{\theta}}\left(1 - y\left(\vec{\theta}\cdot\vec{x}\right)\right) = -y\,\nabla_{\vec{\theta}}\left(\vec{\theta}\cdot\vec{x}\right) = -y\,\vec{x}$$

Substituting into the SGD step, the two minus signs cancel:

$$\boxed{\vec{\theta}^{(k+1)} = \vec{\theta}^{(k)} + \eta_k\, y^{(i)}\vec{x}^{(i)} \quad \text{if } y^{(i)}\left(\vec{\theta}^{(k)}\cdot\vec{x}^{(i)}\right) < 1}$$

> **Compare to the perceptron update rule.** Two differences, and they're the entire lecture in miniature:
>
> | | Perceptron | SGD + hinge loss |
> |---|---|---|
> | Updates when | $y(\vec{\theta}\cdot\vec{x}) \leq 0$ — only on **mistakes** | $y(\vec{\theta}\cdot\vec{x}) < 1$ — also on **correct-but-low-margin** points |
> | Step size | fixed at $1$, not tunable | $\eta_k$, a **tunable hyperparameter** |
>
> The perceptron is (almost) SGD on hinge loss with $\eta = 1$ and the threshold moved from $1$ to $0$. Demanding margin $\ge 1$ instead of $> 0$ is what makes it work on non-separable data.

---

## 9. Practical Notes

### 9.1 Why shuffle the points?

- **Removes the effect of any unintended / artificial ordering of the data** — e.g. a file sorted by class, all the $+$ followed by all the $-$, which would drag $\vec\theta$ back and forth in long biased runs.
- **Faster convergence.**

### 9.2 Convergence criteria

In the perceptron we checked "are all points correctly classified?" That won't do here: **$\mathcal{E}_N$ may never reach $0$**, since the data may not be linearly separable. Similarly, **$R_N$ won't go to zero**.

Instead, look at **change** rather than absolute value — stop when either

- **$R_N(\vec{\theta})$ changes by less than $\epsilon$**, or
- **$\vec{\theta}$ changes by less than $\epsilon$**: $\left\|\vec{\theta}^{(k+1)} - \vec{\theta}^{(k)}\right\| < \epsilon$.

(In practice a max-iteration cap is also standard, so a non-converging run still terminates.)

### 9.3 How do we set the learning rate $\eta$?

| $\eta$ | Effect |
|---|---|
| **too small** | slow — many tiny steps to get anywhere |
| **too large** | **overshoot** — bounce across the minimum, may never converge |

$\eta$ is **a hyperparameter that can and should be tuned.** It may also be set as a **function of $k$**, decaying as you approach the minimum:

$$\eta_k = \frac{1}{k+1} \qquad \text{(just one example — other schedules exist)}$$

---

## Appendix (from the slides' margin notes)

**A good learning rate ensures convergence — the Robbins–Monro conditions:**

$$\sum_{k} \eta_k = \infty \qquad \text{and} \qquad \sum_{k} \eta_k^2 < \infty$$

Intuition: the first says the steps sum to enough total distance to reach the minimum from anywhere; the second says they shrink fast enough to settle rather than rattle around forever. Note $\eta_k = \frac{1}{k+1}$ satisfies both.

With an appropriate learning rate, **if $R_N(\vec{\theta})$ is convex, SGD converges to the global minimum almost surely.**

**SGD is a general algorithm** that can be applied to **non-convex** functions as well — in which case it converges to a **local** minimum. (This is what actually happens when training neural networks.)

**Technically $R_N$ with hinge loss is not everywhere differentiable**, since it's piecewise linear. What do we do?

- **Where differentiable** — no problem.
- **Where subdifferentiable** — **choose any gradient around the kink.** (At $z=1$ the valid subgradients run from $-y\vec{x}$ to $\vec{0}$; picking either endpoint is fine, which is why the $<$ vs. $\leq$ in the update condition doesn't matter in practice.)

---

## Quick Reference

```
margin           z = y(θ⃗ · x⃗)          sign → right/wrong,  |z| ∝ distance from boundary

E_N(θ⃗)  = (1/N) Σ 1[ y⁽ⁱ⁾(θ⃗ · x⃗⁽ⁱ⁾) ≤ 0 ]        training error      (NP-hard, non-convex,
R_N(θ⃗)  = (1/N) Σ loss(y⁽ⁱ⁾, θ⃗ · x⃗⁽ⁱ⁾)           empirical risk       discontinuous)

loss_0-1(z) = 1[z ≤ 0]      → R_N = E_N.  not continuous, not differentiable, not convex
loss_h(z)   = max{0, 1 − z} → hinge.      continuous, differentiable*, CONVEX

GRADIENT DESCENT                        minimize f: R^d → R
  θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ − η_k ∇_θ f(θ⃗)|_{θ⃗⁽ᵏ⁾}      gradient points UPHILL ⇒ step against it

STOCHASTIC GRADIENT DESCENT             shuffle, then one update per data point
  θ⃗⁽ᵏ⁺¹⁾ = θ⃗⁽ᵏ⁾ − η_k ∇_θ loss(y⁽ⁱ⁾, θ⃗ · x⃗⁽ⁱ⁾)

SGD + HINGE LOSS
  if y⁽ⁱ⁾(θ⃗ · x⃗⁽ⁱ⁾) ≥ 1 :  loss = 0, gradient = 0⃗,  no update
  else                  :  ∇ = −y⁽ⁱ⁾x⃗⁽ⁱ⁾  ⇒  θ⃗ ← θ⃗ + η_k y⁽ⁱ⁾ x⃗⁽ⁱ⁾
  vs. perceptron: threshold 1 instead of 0, and a tunable η

key identity     ∇_θ (θ⃗ · x⃗) = x⃗
1-D warm-up      f(θ) = θ² − 10θ,  f′ = 2θ − 10,  θ* = 5
                 θ⁽ᵏ⁺¹⁾ = θ⁽ᵏ⁾ − η(2θ⁽ᵏ⁾ − 10)

convergence      |ΔR_N| < ε   or   ‖θ⃗⁽ᵏ⁺¹⁾ − θ⃗⁽ᵏ⁾‖ < ε      (E_N and R_N won't hit 0)
learning rate    too small ⇒ slow;  too large ⇒ overshoot.   e.g. η_k = 1/(k+1)
Robbins–Monro    Σ η_k = ∞  and  Σ η_k² < ∞   ⇒ convex R_N: global min almost surely
                 non-convex ⇒ local min.  * hinge: subdifferentiable at the kink
```
