# CS-334 Machine Learning — Lecture 02: Python Crash Course
**Date:** 09/01/2026
**Demo repo:** https://github.com/shengpu-tang/CS334-demo (`Python_Tutorial.ipynb` → "Open in Colab")

---

## Admin / Logistics

- **In-Class Exercise #1** — reopened on Gradescope; submit if you haven't.
- **In-Class Exercise #2** — due today, submission closes **11:59pm**.
- **HW1** — posted, due **Sun 9/13** (two Gradescope entries: HW1-Written and HW1-Code).
- Gradescope course: `https://www.gradescope.com/courses/1341911`, entry code **X8BBB7**.
- Calculus and linear algebra refresher notes: Canvas → Files → Readings.
- Slides on Canvas. `iris_cs334.csv` for Exercise 4 is on Canvas under In-Class Exercises.

---

## 1. Why Python

| Pro | Con |
|---|---|
| Easy to learn | Interpreted, not compiled — can be slow |
| Elegant, concise syntax | |
| Huge scientific computing ecosystem | |

Python was ~19.6% of job postings in the cited 2022–23 survey (2nd behind JS/TS). Environment for this course: **Jupyter / Google Colab**. NumPy, pandas, matplotlib are pre-installed on Colab; locally you'd `pip install` them.

---

## 2. Variables and Types

A variable puts data in memory and gives it a name. Created with the assignment operator `=`.

```python
x = 3          # int
pi = 3.14      # float
day = "Tuesday"  # str — a sequence of characters
is_tuesday = True  # bool — note the capitalization: True / False
# comment: not executed
```

Check a type with `type(x)`. Get length of a sequence with `len(hello)`.

Compound assignment: `x += 1`, `x *= 2`.

---

## 3. Operators

| Operation | Python | Example |
|---|---|---|
| Addition | `+` | `1 + 2` |
| Subtraction | `-` | `2 - 1` |
| Multiplication | `*` | `2 * 4` |
| Division | `/` | `4 / 2` |
| Exponentiation | `**` | `x ** 3` |
| Modulo | `%` | `4 % 3` |
| Equality | `==` | `2 == 3` |
| Comparison | `<`, `<=`, `>`, `>=` | `2 < 3` |
| Logical | `and`, `or`, `not` | `x < 5 and x > 2` |

`!=` acts as logical XOR on booleans (`t != f` → `True`).

### Two kinds of divide

```python
print(17 / 3)   # 5.666666666666667  — true division, always returns float
print(17 // 3)  # 5                  — floor division, returns int
print(17 % 3)   # 2                  — remainder
```

> **Exercise 2.1 — Float division.** `11.0 // 2.5`
> Floor division on floats still returns a **float**: `4.0`, not `4`. The `//` operator floors the result but the type is promoted to float because an operand is a float.

### Strings

```python
hello = 'hello'   # single or double quotes, doesn't matter
world = "world"
hw = hello + ' ' + world   # concatenation
print(hw)   # hello world
```

> **Exercise 2.2 — "Adding" a string and a number.** `s = 'CS'; x = 334`
> `s + x` raises `TypeError` — Python does not implicitly coerce. Fix with `s + str(x)` → `'CS334'`, or use an f-string: `f'{s}{x}'`. Note `s * 3` *does* work (repetition).

Full list of string methods: https://docs.python.org/3/library/string.html

---

## 4. Containers

### Lists

```python
x = [1, 2, 3, 'a', 'b', 'c']  # lists can hold mixed types
x[0]        # zero-indexed
x[-1]       # negative indices count from the end
x[0] = 100  # mutable — modify in place
x.append('element')
x.pop()             # removes and returns last item
'element' in x      # membership test
x[6]                # IndexError if out of range
```

Concatenate with `+`: `[1,2,3] + ['A','B','C']`.

**Slicing** — `x[start:stop]`, right-exclusive:

```python
x = [1, 2, 3, 4, 5]
x[2:]    # [3, 4, 5]
x[:3]    # [1, 2, 3]
x[2:5]   # [3, 4, 5]
x[2:5] = [-3, -4, -5]   # slice assignment modifies in place
```

### ⚠️ Common mistake: copying lists

```python
first_list = [1, 2, 3]
second_list = first_list   # NOT a copy — both names point to the same object
second_list[2] = 100
print(first_list)   # [1, 2, 100]  ← surprise
```

Fix — take a full slice (shallow copy):

```python
second_list = first_list[:]   # or list(first_list), or first_list.copy()
second_list[2] = 100
print(first_list)    # [1, 2, 3]
print(second_list)   # [1, 2, 100]
```

**Key idea:** assignment binds a *name* to an *object*. It never copies. This bites you on every mutable type.

### Dictionaries

Like lists, but indexed by (hashable) keys instead of integers.

```python
a = {}                  # empty dict
a["key"] = "value"
print(a)                # {'key': 'value'}
print(a["key"])         # value
```

> **Exercise 2.3 — Copying dictionaries.**
> ```python
> dct1 = {'hello': 'world!', "it's!": 'me!'}
> dct2 = dct1          # ← same aliasing bug as lists
> dct2["it's!"] = 'you!'
> ```
> Both dicts change. Fix with `dct2 = dct1.copy()` or `dct2 = dict(dct1)`. Note `dct1[:]` does **not** work for dicts — slicing isn't defined. For nested structures you'd need `copy.deepcopy`.

---

## 5. Functions

```python
def hours_and_minutes(minutes):
    n_hours = minutes // 60
    n_minutes = minutes % 60
    return n_hours, n_minutes
```

Anatomy:
- `def` — about to define a function
- `hours_and_minutes` — name
- `(minutes)` — input arguments
- indented block — body. **Whitespace/tabs define scope in Python.**
- `return` — what comes out; returning multiple values gives back a **tuple**

```python
print(hours_and_minutes(181))  # (3, 1)
print(n_minutes)               # NameError — local to the function, not visible outside
```

---

## 6. Control Flow

### if / elif / else

```python
if day == 'Tuesday' or day == 'Thursday':
    print('Time to Machine Learn!')
elif day in ['Saturday', 'Sunday']:
    print('The Weekend!')
else:
    print('Just Another Day :/')
```

- Condition must evaluate to `True`/`False`
- **Colons** at the end of each header line
- `elif` blocks evaluated in order; `else` runs if nothing matched

### while

```python
i = 0
while i < 3:
    print(i)
    i += 1
print(i)     # 0 1 2 then 3
```

Note the loop variable survives after the loop, and prints `3` at the end — the condition is checked *before* each pass.

### for

```python
animals = ['cat', 'dog', 'monkey']

for animal in animals:      # iterate over items directly (preferred)
    print(animal)

for i in range(len(animals)):   # iterate over indices
    print(animals[i])
```

### List comprehension

```python
x = [1, 2, 3]

# long form
y = []
for item in x:
    y.append(item * item)

# comprehension — same result, one line
y = [item * item for item in x]
print(y)   # [1, 4, 9]
```

---

## 7. Modules / Packages

A module is a collection of variables, functions, and classes. Also called library or package.

```python
import math;            math.sqrt(16)
from math import sqrt;  sqrt(16)
import math as m;       m.sqrt(16)
```

---

## 8. NumPy

Tensor/matrix library. Like lists, but n-dimensional and much faster (contiguous typed memory, vectorized C loops).

```python
import numpy as np
x = np.array([1, 2, 3])              # vector, shape (3,)
A = np.array([[1,0,0], [0,1,1]])     # 2x3 matrix
A_ = A.T                             # transpose → 3x2
```

### Element-wise operations (broadcasting)

```python
x + 1     # [2 3 4]
x * 2     # [2 4 6]
x ** 2    # [1 4 9]
x > 1     # [False  True  True]   ← boolean mask
x == 1    # [ True False False]
```

`*` on arrays is **element-wise**, not matrix multiplication.

### Dot / matrix products

Three equivalent spellings:

```python
x @ x        # 14
np.dot(x, x) # 14
x.dot(x)     # 14

A @ x        # [1 5]
np.dot(A, x) # [1 5]
A.dot(x)     # [1 5]
```

### Summary statistics

```python
np.sum(x)    # 6
np.mean(x)   # 2.0
np.std(x)    # 0.8164965809277263

np.sum(A,  axis=0)   # [1 1 1]   ← axis=0 collapses rows (column-wise result)
np.mean(A, axis=0)   # [0.5 0.5 0.5]
np.std(A,  axis=0)   # [0.5 0.5 0.5]
```

Mnemonic: `axis=k` is the axis that **disappears** from the output shape.

### Exercises Q3

Setup:
```python
rng = np.random.default_rng(334)
x1 = rng.random(4)                    # shape (4,)
x2 = rng.integers(0, 20, size=(5,4))  # shape (5,4)
x3 = rng.random((5,4,3))              # shape (5,4,3)
x4 = rng.integers(0, 10, size=(5,4))  # shape (5,4)
```

> **Q3.1 — `.shape` vs `.size`.** `.shape` is the tuple of dimensions; `.size` is the total number of elements (the product of the shape). E.g. `x3.shape == (5,4,3)` but `x3.size == 60`. Also `x3.ndim == 3`, and `len(x3)` gives only the first dimension (5).

> **Q3.2 — Multiplication.** `x2 * x4` is element-wise (same shape → shape (5,4)). `x2 @ x4` fails: inner dimensions (5,4)·(5,4) don't line up — you'd need `x2 @ x4.T` → (5,5) or `x2.T @ x4` → (4,4). Broadcasting also lets `x2 * x1` work, since (5,4) and (4,) align on the trailing axis.

> **Q3.3 — dtype gotcha.**
> ```python
> arr = np.array([1,2,3])   # dtype inferred as int64
> arr[0] = 1.5
> print(arr)                # [1 2 3]  ← silently truncated!
> ```
> NumPy arrays are **homogeneously typed and fixed dtype**. Assigning a float into an int array truncates toward zero without warning. Fix: create it as float from the start — `np.array([1,2,3], dtype=float)` or `np.array([1.,2.,3.])` — or `arr = arr.astype(float)` first. This is a common silent-bug source in ML code.

---

## 9. Pandas

A DataFrame is a 2D table like a NumPy array, except **rows/columns have names** and it can **hold non-numeric data**.

```python
import pandas as pd
df = pd.DataFrame(
    [[1, 1, 'odd'], [2, 4, 'even'],
     [3, 9, 'odd'], [4, 16, 'even']],
    columns=['x', 'x^2', 'parity'])
```

|   | x | x^2 | parity |
|---|---|-----|--------|
| 0 | 1 | 1   | odd    |
| 1 | 2 | 4   | even   |
| 2 | 3 | 9   | odd    |
| 3 | 4 | 16  | even   |

### Boolean (logical) slicing

```python
df[df['parity'] == 'odd']            # rows 0 and 2
```

### Chaining slice + summary statistic

```python
df[df['parity'] == 'odd']['x'].mean()   # 2.0
```

Read the pattern as: build a boolean mask → filter rows → select a column → aggregate.

---

## 10. Matplotlib

```python
import matplotlib.pylab as plt

x = np.arange(5)
y = x ** 2
plt.plot(x, y)
plt.show()
```

Styling — format string is `color + marker + linestyle`:

```python
x = np.linspace(0, 1, 10)
plt.plot(x, x**2,     'ro:', label="parabola")   # red, circles, dotted
plt.plot(x, np.sin(x),'gx--', label="sine")      # green, x-marks, dashed
plt.xlabel('some numbers')
plt.ylabel('some more numbers')
plt.legend()
plt.show()
```

Scatter:

```python
x = np.random.uniform(1, 1000, 1000)
y = np.log(x) + np.random.normal(0, .3, 1000)
plt.scatter(x, y, s=5)   # s = marker size
plt.show()
```

---

## 11. Exercise 4 — Data Exploration (iris)

Download `iris_cs334.csv` from Canvas (In-Class Exercises folder), upload to Colab with the **upload button** in the file picker on the left sidebar. Note: Colab's filesystem is ephemeral — files vanish when the runtime disconnects.

```python
df = pd.read_csv('iris_cs334.csv')
df   # 100 rows x 5 columns
```

Columns: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `variety` (Setosa / Virginica — note this trimmed version has only 2 of the usual 3 classes).

**Q4.1 Summary statistics**
```python
df['petal_length'].median()
df[df['variety'] == 'Virginica']['sepal_width'].mean()
df[df['variety'] == 'Setosa']['sepal_length'].std()
```

**Q4.2 Scatterplot** — petal_length vs petal_width, colored by variety. The two classes separate cleanly, which is the whole reason iris is the standard toy classification dataset.

**Q4.3 Boxplot** — sepal length grouped by variety (x = variety, y = sepal_length); Virginica sits noticeably higher than Setosa, though with more overlap than the petal features show.

---

## Things worth remembering

1. **Assignment never copies.** `b = a` aliases. Use `a[:]`, `.copy()`, or `copy.deepcopy` for nested structures.
2. **`/` vs `//`.** True division always returns float; floor division returns int only if both operands are ints.
3. **Indentation is syntax**, not style. It defines scope.
4. **NumPy arrays have a fixed dtype** — assigning a float into an int array truncates silently.
5. **`*` is element-wise, `@` is matrix multiply.** Mixing them up is the most common NumPy bug.
6. **`axis=k` collapses axis k.** `axis=0` → column-wise stats, `axis=1` → row-wise.
7. Pandas filter idiom: `df[df[col] == val][other_col].agg()`.
