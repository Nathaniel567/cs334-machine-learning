# In-Class Exercise 03 — Answers

## Q1 Automatic Credit

### Q1.1 Attestation
Yes

---

## Q2 ML Terminology

*Setup: building a spam classifier from 100 tagged emails (subject + content), viewed as binary classification.*

### Q2.1 Feature
- Number of suspicious words/phrases in the email
- Whether the sender's domain is trusted

### Q2.2 Feature Vector
For a given email, the feature vector is:

$$\vec{x} = (\#\text{suspicious words/phrases},\ \text{domain trusted?})$$

i.e. `x = (# of suspicious words or phrases, domain is trusted)`.

### Q2.3 Feature Space
$\mathbb{R}^2$

(Strictly, the first coordinate is a nonnegative count and the second is binary — e.g. $\{0,1,2,\dots\}\times\{0,1\}$ — but $\mathbb{R}^2$ is the standard simplification used once features are numeric.)

### Q2.4 Labels/Classes
- Positive class: spam
- Negative class: not spam

### Q2.5 Examples
100

---

## Q3 Linear Decision Boundaries

*Setup: for each dataset, find $\vec{\theta} \in \mathbb{R}^2$ such that $\vec{\theta}^\top \vec{x} = 0$ is a decision boundary through the origin achieving zero training error (red "x" = positive, blue "o" = negative), or say N/A if impossible.*

### Q3.1 Parameter Vectors
- (a) $\vec{\theta} = (-1, 0)$
- (b) N/A
- (c) $\vec{\theta} = (1, 2)$
- (d) N/A

### Q3.2 Uniqueness
**No** — the solutions in (a) and (c) are not unique.

**Why:** the decision boundary is the set $\{\vec{x} : \vec{\theta}^\top \vec{x} = 0\}$, and the classification rule uses the *sign* of $\vec{\theta}^\top \vec{x}$. Scaling $\vec{\theta}$ by any $c > 0$ doesn't change either one:
$$\text{sign}\big((c\vec{\theta})^\top \vec{x}\big) = \text{sign}\big(c \cdot (\vec{\theta}^\top \vec{x})\big) = \text{sign}\big(\vec{\theta}^\top \vec{x}\big) \quad \text{for } c>0$$
So for (a), $(-2, 0)$, $(-0.5, 0)$, etc. all work just as well as $(-1, 0)$ — infinitely many correct $\vec{\theta}$ point in the same *direction* but differ in magnitude. Same for (c) with any positive multiple of $(1, 2)$.

(For (b) and (d), uniqueness is moot since no solution exists at all — the datasets aren't linearly separable by a boundary through the origin.)

---

*Verified 09/08/2026 against Prof. Tang's own solutions, shown in Lecture 04 (`lectures/slides/Lecture-04-Perceptron.pdf`, p.9). All four answers are confirmed:*

- *(a) He drew the boundary $x_2 = x_1$ with $\vec{\theta}$ "pointing toward the upper left." The submitted $(-1,0)$ also achieves zero training error — the reds sit at $x_1 \in \{-2,-1\}$ and the blues at $x_1 \in \{1,2\}$, so the $x_2$-axis separates them cleanly. Any $\vec{\theta}$ in the cone between $(-1,0)$ and $(0,1)$ works.*
- *(b) N/A — his reasoning: **a point sitting at the origin is always misclassified**, since $\vec{\theta}\cdot\vec{0}=0$ for every $\vec{\theta}$ and $0$ counts as wrong by convention.*
- *(c) $\vec{\theta}=[1,2]^T$ — matches exactly. His reasoning: the boundary slope must fall between $0$ and $-1$, so $\theta_1/\theta_2 \in (0,1)$ and $\vec{\theta}$ points toward the upper right.*
- *(d) N/A — XOR-style layout, not separable by any line through the origin.*
