# CS-334 Machine Learning — Lecture 04: Perceptron
**Date:** 09/08/2026 · **Instructor:** Prof. Shengpu Tang
**Slides:** `lectures/slides/Lecture-04-Perceptron.pdf`
**Prereq:** `Lecture-03-Linear-Classification.md`

---

## Admin / Logistics

- **HW1** due **Sun 11:59pm** on Gradescope. Autograder accepts multiple submissions.
- **In-Class Exercises are graded** — check your score and file a regrade request if something's off.
- The perceptron **with offset** will be implemented in **HW2**.

---

## 1. ML Terminology, Concretely: Loan Credit Risk Prediction

The slides build up the vocabulary one layer at a time. Final form:

| Term | In this example | Symbol |
|---|---|---|
| **Features** | age: int in $[0,120]$; employment status: {student, employed, unemployed} | $x_1 \in \{0..120\}$, $x_2 \in \{\text{student, employed, unemployed}\}$ |
| **Feature vector** | Given applicant $i$, ask their age and employment status | $\vec{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]$ |
| **Feature space** | The set of *all possible* combinations | $\{0..120\} \times \{\text{student, employed, unemployed}\}$ |
| **Labels / classes** | approve vs. not approve | $y = +1$ approve, $y = -1$ not approve |

Two things worth noticing:

- **Feature space is a Cartesian product** of the individual feature domains — it's every combination, not just the ones you observed.
- **$x_2$ here is categorical, not numeric.** A linear classifier needs numbers, which is exactly why HW1 Q4 asks you to **one-hot encode** the categorical `species` column of the penguins dataset. The terminology and the homework are the same idea.

---

## 2. Recap: Linear Classifier Through the Origin

$$h(\vec{x};\vec{\theta}) = \mathrm{sign}(\vec{\theta}\cdot\vec{x}) = \begin{cases}
+1 & \text{if } \vec{\theta}\cdot\vec{x} > 0\\
0^{*} & \text{if } \vec{\theta}\cdot\vec{x} = 0\\
-1 & \text{if } \vec{\theta}\cdot\vec{x} < 0
\end{cases} \qquad \vec{x}\in\mathbb{R}^d,\ \vec{\theta}\in\mathbb{R}^d$$

$^{*}$ When making predictions the output **must be binary**, so we can map $0$ to either $+1$ or $-1$ — it's a free convention choice. (Separately, for *scoring* training error, a point on the boundary counts as **misclassified**; see §3.)

$\vec{\theta}$ goes by several names, all the same object: **parameter vector · weight vector · normal vector · model coefficients**.

**It all comes down to the dot product $\vec{\theta}\cdot\vec{x}$.** $\vec{\theta}$ defines a hyperplane in $d$-dimensional space through the origin:

- $\vec{\theta}\cdot\vec{x} > 0$ → positive side (the side $\vec{\theta}$ points toward)
- $\vec{\theta}\cdot\vec{x} = 0$ → the decision boundary itself
- $\vec{\theta}\cdot\vec{x} < 0$ → negative side

> **What does this look like in 3D?** The boundary is a 2D plane through the origin, with $\vec{\theta}$ sticking out perpendicular to it. In $d$ dimensions it's a $(d-1)$-dimensional hyperplane.

---

## 3. Recap: Training Error

$$\mathcal{E}_N(\vec{\theta}) = \underbrace{\frac{1}{N}\sum_{i=1}^{N}}_{\text{fraction of training examples}} \mathbb{1}\underbrace{\left[y^{(i)}\left(\vec{\theta}\cdot\vec{x}^{(i)}\right) \leq 0\right]}_{\text{where the classifier predicts the wrong label}}$$

**Points on the decision boundary are considered misclassified** — hence $\leq$, not $<$.

---

## 4. Solutions to In-Class Exercise #3 (Q3)

The professor worked the four datasets from last lecture. Each asks: find $\vec{\theta}$ giving a zero-error boundary **through the origin**, or answer N/A.

| | Answer | Reasoning |
|---|---|---|
| **(a)** | any $\vec{\theta}$ **pointing toward the upper left** | reds sit upper-left, blues lower-right; e.g. $\vec{\theta} = (-1,1)$, and $(-1,0)$ or $(0,1)$ also give zero error |
| **(b)** | **N/A** | **not possible — a point sitting at the origin is always misclassified.** $\vec{\theta}\cdot\vec{0} = 0$ for every $\vec{\theta}$, and $0$ counts as wrong |
| **(c)** | $\vec{\theta} = [1,2]^T$ (any positive multiple) | boundary slope must be between $0$ and $-1$ ⟹ $\theta_1/\theta_2 \in (0,1)$ ⟹ $\vec{\theta}$ points upper-right |
| **(d)** | **N/A** | XOR-style layout — no line through the origin separates it |

> **(b) is the one to remember.** The origin is a permanent blind spot for a classifier without an offset. This is a big part of *why* §6 exists.

### After drawing a decision boundary, how do you actually find $\vec{\theta}$?

- **Method 1 — eyeball it.** $\vec{\theta}$ is perpendicular to the boundary; pick the perpendicular direction that points into the **positive** region.
- **Method 2 — solve it.** Pick any point on the boundary; it satisfies $\vec{\theta}\cdot\vec{x} = 0$. Solve for the ratio $\theta_1/\theta_2$, then pick the sign so $\vec{\theta}$ points to the positive side.

Recall length never matters, only direction — so quoting a ratio is a complete answer.

---

## 5. The Perceptron Algorithm

### 5.1 The goal, and why it's hard

$$\text{find } \vec{\theta}^{*} = \arg\min_{\vec{\theta}} \mathcal{E}_N(\vec{\theta})$$

**How?**
- **In general this is not easy to solve — it's NP-hard.** ($\mathcal{E}_N$ is a sum of step functions: piecewise constant, zero gradient almost everywhere, so gradient descent has nothing to descend.)
- **For now, consider a special case — linearly separable data.**

### 5.2 Linear separability

> **Definition.** Training examples $\mathcal{D} = \{(\vec{x}^{(i)}, y^{(i)})\}_{i=1}^{N}$ are **linearly separable through the origin** if there exists a parameter $\vec{\theta}$ such that
> $$y^{(i)}\left(\vec{\theta}\cdot\vec{x}^{(i)}\right) > 0 \quad \forall i = 1 \dots N$$

Note the **strict $>$** — no point may sit on the boundary. This is exactly the condition "$\mathcal{E}_N(\vec{\theta}) = 0$ is achievable."

### 5.3 The algorithm

**Mistake-driven:** start with $\vec{\theta} = \vec{0}$ (the zero vector) and update $\vec{\theta}$ only to correct mistakes.

```
k = 0,  θ⁽⁰⁾ = 0⃗
while not all points are correctly classified:
    for i = 1 … N:
        if y⁽ⁱ⁾ (θ⁽ᵏ⁾ · x⃗⁽ⁱ⁾) ≤ 0:        ← a mistake (boundary counts)
            θ⁽ᵏ⁺¹⁾ = θ⁽ᵏ⁾ + y⁽ⁱ⁾ x⃗⁽ⁱ⁾      ← update rule
            k = k + 1
```

- $\vec{\theta}^{(k)}$ denotes the parameters **after $k$ mistakes** — $k$ counts *mistakes*, not iterations or epochs.
- **Parameters are updated only if we make a mistake.** Correctly classified points are skipped entirely.
- Why does $\vec{\theta} \mathrel{+}= y\vec{x}$ help? It rotates $\vec{\theta}$ toward $\vec{x}$ when $y=+1$ and away when $y=-1$ — nudging the boundary so that this point lands on the right side. See §7 for the formal reason it's a *nudge* and not a guaranteed fix.

### 5.4 Worked example

Two points: $\vec{x}^{(1)} = [6,6]^T,\ y^{(1)} = +1$ and $\vec{x}^{(2)} = [9,1]^T,\ y^{(2)} = -1$.

| Step | Check | Result |
|---|---|---|
| Init | — | $\vec{\theta}^{(0)} = [0,0]^T$ |
| $i=1$ | $y^{(1)}(\vec{\theta}^{(0)}\cdot\vec{x}^{(1)}) = 0 \leq 0$ → mistake | $\vec{\theta}^{(1)} = [0,0] + [6,6] = [6,6]^T$ |
| $i=2$ | $y^{(2)}(\vec{\theta}^{(1)}\cdot\vec{x}^{(2)}) = -(54+6) = -60 \leq 0$ → mistake | $\vec{\theta}^{(2)} = [6,6] - [9,1] = [-3,5]^T$ |

Final check with $\vec{\theta}^{(2)} = [-3,5]^T$: margins are $+12$ and $+22$ — both positive, so **the classifier correctly classifies each point after seeing them once**, and the algorithm converges after **one (outer) loop**.

*(Note the very first update: $\vec{\theta}^{(0)} = \vec{0}$ gives margin exactly $0$, which counts as a mistake — that's how the algorithm gets off the ground at all.)*

> **Theorem.** The perceptron algorithm **converges after a finite number of mistakes** if the training examples are **linearly separable (through the origin)**.
>
> **However, the solution may not be unique**, and each point may be seen more than once or not at all.

---

## 6. What If It's *Not* Linearly Separable Through the Origin?

Two cases:

- **Easy case: the data is separable, just not by a line through the origin** → **add an offset.** (This lecture.)
- **Hard case: no line separates it at all** (XOR-style) → **next lecture.**

### 6.1 Linear classifier with offset

$$h(\vec{x};\vec{\theta}, b) = \mathrm{sign}(\vec{\theta}\cdot\vec{x} + b), \qquad \vec{x},\vec{\theta}\in\mathbb{R}^d,\ \boxed{b \in \mathbb{R}}$$

$b$ is the **offset**, aka **intercept**. It's a scalar, not a vector.

- The new boundary $\vec{\theta}\cdot\vec{x} + b = 0$ **does not pass through the origin** (when $b \neq 0$).
- It is **parallel** to $\vec{\theta}\cdot\vec{x} = 0$:
  - they have the same normal vector $\vec{\theta}$;
  - if they intersected, some point would satisfy $\vec{\theta}\cdot\vec{x} + b = \vec{\theta}\cdot\vec{x}$, which is only possible when $b = 0$.
- The **signed distance** from the origin-boundary to the new one is $\dfrac{-b}{\|\vec{\theta}\|}$. **→ this shows up in HW1.**

### 6.2 Deriving $-b/\|\vec{\theta}\|$

Worth working through by hand — HW1 Q1(c) is the same technique.

1. Pick $\vec{x}^{(1)}$ on the **old** boundary: $\vec{\theta}\cdot\vec{x}^{(1)} = 0$.
2. Pick $\vec{x}^{(2)}$ on the **new** boundary: $\vec{\theta}\cdot\vec{x}^{(2)} = -b$.
3. Let $\vec{v} = \vec{x}^{(2)} - \vec{x}^{(1)}$.
4. Project $\vec{v}$ onto the direction of $\vec{\theta}$: $\ \mathrm{proj}_{\vec{\theta}}\vec{v} = \left(\vec{v}\cdot\frac{\vec{\theta}}{\|\vec{\theta}\|}\right)\frac{\vec{\theta}}{\|\vec{\theta}\|}$, where $\frac{\vec{\theta}}{\|\vec{\theta}\|}$ is the **unit vector** in the direction of $\vec{\theta}$.
5. The signed distance is the scalar coefficient:

$$\vec{v}\cdot\frac{\vec{\theta}}{\|\vec{\theta}\|} = \frac{(\vec{x}^{(2)} - \vec{x}^{(1)})\cdot\vec{\theta}}{\|\vec{\theta}\|} = \frac{\vec{x}^{(2)}\cdot\vec{\theta} - \vec{x}^{(1)}\cdot\vec{\theta}}{\|\vec{\theta}\|} = \frac{-b - 0}{\|\vec{\theta}\|} = \boxed{\frac{-b}{\|\vec{\theta}\|}}$$

So $b < 0$ shifts the boundary in the **$+\vec{\theta}$** direction; $b > 0$ shifts it the other way.

**Separability with offset:** $\mathcal{D}$ is linearly separable with offset if there exist $\vec{\theta}$ **and** $b$ such that $y^{(i)}(\vec{\theta}\cdot\vec{x}^{(i)} + b) > 0\ \ \forall i = 1\dots N$.

### 6.3 Perceptron with offset

```
k = 0,  θ⁽⁰⁾ = 0⃗,  b⁽⁰⁾ = 0
while not all points correct (with θ⁽ᵏ⁾, b⁽ᵏ⁾):
    for i = 1 … N:
        if y⁽ⁱ⁾ (θ⁽ᵏ⁾ · x⃗⁽ⁱ⁾ + b⁽ᵏ⁾) ≤ 0:
            θ⁽ᵏ⁺¹⁾ = θ⁽ᵏ⁾ + y⁽ⁱ⁾ x⃗⁽ⁱ⁾
            b⁽ᵏ⁺¹⁾ = b⁽ᵏ⁾ + y⁽ⁱ⁾            ← the only new line
            k = k + 1
```

**→ will implement in HW2.**

**How was this derived?** It's not a new algorithm — it's the *same* algorithm on augmented vectors. Augment each input with a constant 1 and fold $b$ into the parameter vector:

$$\vec{x}' = [1, \vec{x}]^T, \qquad \vec{\theta}' = [b, \vec{\theta}]^T \quad\Longrightarrow\quad \vec{\theta}'\cdot\vec{x}' = b + \vec{\theta}\cdot\vec{x}$$

Then apply the same update rule. The update to the first coordinate is $y^{(i)} \cdot 1 = y^{(i)}$, which is exactly $b \mathrel{+}= y^{(i)}$. **This augmentation trick recurs all semester** — it's how "with offset" is handled almost everywhere.

---

## 7. Question: Can the Algorithm Undershoot?

**Yes.** Meaning: after updating on a mistake, that *same* example can still be misclassified.

Suppose we make a mistake on $\vec{x}^{(i)}$ and update: $\vec{\theta}^{(k+1)} = \vec{\theta}^{(k)} + y^{(i)}\vec{x}^{(i)}$. Now re-classify that exact same example and ask when it's still wrong:

$$y^{(i)}\left(\vec{\theta}^{(k+1)}\cdot\vec{x}^{(i)}\right) \leq 0$$
$$y^{(i)}\left(\left(\vec{\theta}^{(k)} + y^{(i)}\vec{x}^{(i)}\right)\cdot\vec{x}^{(i)}\right) \leq 0$$
$$y^{(i)}\left(\vec{\theta}^{(k)}\cdot\vec{x}^{(i)}\right) + \underbrace{y^{(i)}y^{(i)}}_{=\,1}\underbrace{\left(\vec{x}^{(i)}\cdot\vec{x}^{(i)}\right)}_{=\,\|\vec{x}^{(i)}\|^2} \leq 0$$
$$\underbrace{y^{(i)}\left(\vec{\theta}^{(k)}\cdot\vec{x}^{(i)}\right)}_{\text{previous margin}} + \underbrace{\|\vec{x}^{(i)}\|^{2}}_{\text{the improvement}} \leq 0$$

**Reading of the result:** each update improves that example's margin by exactly $\|\vec{x}^{(i)}\|^{2}$ — always positive, so it always moves in the right direction, but by a **fixed, finite amount**. **So if $\|\vec{x}\|^2$ is small, we may undershoot**: the margin was more negative than $\|\vec{x}^{(i)}\|^2$ can repair in one step. The point stays wrong and gets corrected on a later pass. (Convergence still holds — hence "may be seen more than once" in the theorem.)

Two consequences worth carrying forward: the **scale of your features matters** (small-magnitude features ⟹ tiny updates ⟹ more passes), and the update size is **not tunable** — there's no learning rate in the plain perceptron.

**The slide's example:** $y^{(i)} = 1$, $\vec{x}^{(i)} = [-0.5, 0.5]^T$, $\vec{\theta}^{(k)} = [1,1]^T$. Here $\|\vec{x}^{(i)}\|^2 = 0.5$, a small step. Running it: the previous margin is $1(-0.5)+1(0.5) = 0$, which counts as a mistake; after the update $\vec{\theta}^{(k+1)} = [0.5, 1.5]^T$ and the new margin is $0 + 0.5 = 0.5 > 0$ — so with these particular numbers the update *just barely* fixes it. It illustrates how small the correction is; to see an actual undershoot you need a starting margin below $-\|\vec{x}\|^2$, e.g. $\vec{\theta}^{(k)} = [1,-1]^T$ gives margin $-1 \to -0.5$, still wrong.

---

## Quick Reference

```
h(x⃗; θ⃗)      = sign(θ⃗ · x⃗)                          through origin
h(x⃗; θ⃗, b)   = sign(θ⃗ · x⃗ + b)                      with offset
E_N(θ⃗)       = (1/N) Σ 1[ y⁽ⁱ⁾(θ⃗ · x⃗⁽ⁱ⁾) ≤ 0 ]      training error
Goal:  θ⃗* = argmin E_N(θ⃗)                            NP-hard in general

linearly separable (origin):  ∃θ⃗ s.t. y⁽ⁱ⁾(θ⃗ · x⃗⁽ⁱ⁾) > 0  ∀i
linearly separable (offset):  ∃θ⃗,b s.t. y⁽ⁱ⁾(θ⃗ · x⃗⁽ⁱ⁾ + b) > 0  ∀i

PERCEPTRON            mistake if  y⁽ⁱ⁾(θ⃗ · x⃗⁽ⁱ⁾ + b) ≤ 0
  θ⃗ ← θ⃗ + y⁽ⁱ⁾ x⃗⁽ⁱ⁾
  b ← b + y⁽ⁱ⁾                    (offset version only)

margin gain per update = ‖x⃗⁽ⁱ⁾‖²   → small ‖x⃗‖ ⇒ may undershoot
augmentation trick:  x⃗′=[1, x⃗]ᵀ,  θ⃗′=[b, θ⃗]ᵀ  ⇒  θ⃗′·x⃗′ = b + θ⃗·x⃗
signed distance origin-plane → offset-plane =  −b / ‖θ⃗‖     (HW1)

Theorem: converges in finitely many mistakes if linearly separable.
         Solution not unique; points may be seen many times or never.
```
