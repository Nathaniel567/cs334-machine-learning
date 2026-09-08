# CS-334 Machine Learning — Lecture 02: Python Crash Course
**Date:** 09/01/2026 · **Instructor:** Prof. Shengpu Tang
**Slides:** `lectures/slides/Lecture-02-Python-Crash-Course.pdf`
**Paired exercise:** In-Class Exercise #2 → `exercises/ex02-python-numpy-pandas/`
**Follow-along notebook:** <https://github.com/shengpu-tang/CS334-demo> (`Python_Tutorial.ipynb` → "Open in Colab")

---

## Admin / Logistics

- **In-Class Exercise #1** — reopened on Gradescope; submit if you haven't.
- **In-Class Exercise #2** — due today, submission closes **11:59pm**.
- **HW1** posted, due **Sun 09/13** (two Gradescope entries: HW1-Written and HW1-Code).
- Gradescope course `1341911`, entry code **X8BBB7**. Slides on Canvas.
- Calculus and linear algebra refresher notes: Canvas → Files → Readings.
- `iris_cs334.csv` for Exercise 4 is on Canvas under In-Class Exercises.

---

## 1. Why Python

| Pro | Con |
|---|---|
| Easy to learn | Interpreted, not compiled — **can be slow** |
| Elegant, concise syntax | |
| Versatile: lots of scientific computing resources | |

Per the cited *devjobsscanner* survey (01-Jan-2022 → 31-May-2023), Python was **603,507 job postings (19.64%)**, second behind JavaScript/TypeScript at 29.80%.

Environment for this course: **Jupyter / Google Colab**. NumPy, pandas, and matplotlib are pre-installed on Colab; locally you'd `pip install` them.

> The "can be slow" con is not a throwaway — it's the entire premise of **HW1 Q3**, which has you time a Python `for` loop against a vectorized NumPy `dot` over sample sizes from 10 to 10M. Keep it in mind through §8.

---

## 2. Variables and Types

A variable puts data in memory and gives it a name. Created with the **assignment operator** `=`.

```python
x = 3              # int   — integers
pi = 3.14          # float — real numbers
day = "Tuesday"    # str   — a sequence of characters
is_tuesday = True  # bool  — note the capitalization: True / False
# comment: notes for programmers, doesn't get executed
```

Assignment evaluates the right side first, so `x = 3 + 3` stores `6`.

Check a type with `type(x)`. Get the length of a sequence with `len(day)`. Compound assignment: `x += 1`, `x *= 2`.

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

On booleans, `!=` behaves as logical **XOR** (`True != False` → `True`).

### 3.1 Two kinds of divide

```python
print(17 / 3)   # 5.666666666666667  — true division, ALWAYS returns float
print(17 // 3)  # 5                  — floor division
print(17 % 3)   # 2                  — remainder
```

> **Exercise 2.1 — Float division.** `print(11.0 // 2.5)` → **`4.0`**, not `4`.
> Floor division *floors the value*, but the **type is still promoted to float** because an operand is a float. `//` does not mean "returns an int."

### 3.2 Strings

```python
hello = 'hello'            # single or double quotes, doesn't matter
world = "world"
hw = hello + ' ' + world   # string concatenation
print(hw)                  # hello world
```

> **Exercise 2.2 — "Adding" a string and a number.** With `s = 'CS'` and `x = 334`:
> `s + x` raises `TypeError: can only concatenate str (not "int") to str` — **Python does not implicitly coerce**.
> Fix with `s + str(x)` → `'CS334'`, or an f-string: `f'{s}{x}'`.
> Note `s * 3` *does* work — that's repetition, giving `'CSCSCS'`.

Full list of string methods: <https://docs.python.org/3/library/string.html>

---

## 4. Containers

### 4.1 Lists

```python
#        index:  0  1  2   3    4    5
x = [1, 2, 3, 'a', 'b', 'c']   # lists can hold mixed types
x[0]         # zero-indexed
x[-1]        # negative indices count from the end
x[3] = 'A'   # mutable — modify in place
x.append(100)       # add item to the end
x.pop()             # removes and returns the last item
'a' in x            # membership test
x[6]                # IndexError if out of range
```

Concatenate with `+`: `[1,2,3] + ['A','B','C']`.

**Slicing** — `x[start:stop]`, **right-exclusive**:

```python
x = [1, 2, 3, 4, 5]
x[2:]    # [3, 4, 5]
x[:3]    # [1, 2, 3]
x[2:5]   # [3, 4, 5]
x[2:5] = [-3, -4, -5]   # slice assignment modifies in place
```

### 4.2 ⚠️ Common mistake: copying lists

```python
first_list = [1, 2, 3]
second_list = first_list   # try to copy — but this is NOT a copy
second_list[2] = 100
print(first_list)    # [1, 2, 100]  ← surprise
print(second_list)   # [1, 2, 100]
```

Fix — take a full slice (shallow copy):

```python
second_list = first_list[:]   # or list(first_list), or first_list.copy()
second_list[2] = 100
print(first_list)    # [1, 2, 3]
print(second_list)   # [1, 2, 100]
```

> **Key idea: assignment binds a *name* to an *object*. It never copies.** `second_list = first_list` makes two names for one list. This bites you on every mutable type — lists, dicts, NumPy arrays, DataFrames.

### 4.3 Dictionaries

Just like lists, but elements can be indexed by **non-integers** (any hashable key).

```python
a = {}                  # empty dict
a["key"] = "value"
print(a)                # {'key': 'value'}
print(a["key"])         # value
```

> **Exercise 2.3 — Copying dictionaries.** Same aliasing bug:
> ```python
> dct1 = {'hello': 'world!', "it's!": 'me!'}
> dct2 = dct1                 # ← aliases, doesn't copy
> dct2["it's!"] = 'you!'      # both dicts now show 'you!'
> ```
> Fix with `dct2 = dct1.copy()` or `dct2 = dict(dct1)`.
> The list trick `dct1[:]` does **not** transfer — it raises `KeyError: slice(None, None, None)`, because `[...]` on a dict is a *key lookup* and a slice object isn't a key.
> For nested structures, neither is enough — you need `copy.deepcopy`.

---

## 5. Functions

```python
def hours_and_minutes(minutes):
    n_hours = minutes // 60
    n_minutes = minutes % 60
    return n_hours, n_minutes
```

Anatomy, as the slides break it down line by line:

| Piece | Role |
|---|---|
| `def` | about to define a function |
| `hours_and_minutes` | name of the function |
| `(minutes)` | input arguments |
| indented block | body — **whitespace/tabs matter, they define the scope** |
| `return` | return value, what we get out |

```python
print(hours_and_minutes(181))  # (3, 1)  ← multiple returns come back as a TUPLE
print(n_minutes)               # NameError — local to the function, invisible outside
```

---

## 6. Control Flow

### 6.1 if / elif / else

```python
if age > 21:
    print("Come in!")
elif age == 20:
    print("Try again next year.")
else:
    print("You've got some time!")
```

- The **condition statement must result in `True`/`False`**.
- **COLONS** at the end of every header line (the slides shout this one).
- `elif` blocks are evaluated in order; `else` runs if no condition was true.

### 6.2 while

```python
i = 0
while i < 3:
    print(i)
    i += 1
print(i)     # prints 0, 1, 2 … then 3
```

The final `3` is the point of the example: the condition is checked **before** each pass, so the loop exits with `i == 3`, and the loop variable **survives after the loop**.

### 6.3 for

```python
animals = ['cat', 'dog', 'monkey']

for animal in animals:            # loop through items in a container directly
    print(animal)

for i in range(len(animals)):     # loop through indices
    print(animals[i])
```

Both print `cat dog monkey`. Prefer the first — iterating items directly is the idiomatic form.

### 6.4 Advanced usage: list comprehension

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

A module is a collection of variables, functions, and classes. Also called a **library** or **package**.

```python
import math;            print(math.sqrt(16))   # 4.0
from math import sqrt;  print(sqrt(16))        # 4.0
import math as m;       print(m.sqrt(16))      # 4.0
```

---

## 8. NumPy

Tensor/matrix operation library. **Lists, but more dimensions, and faster** — contiguous typed memory and vectorized C loops instead of interpreted Python.

```python
import numpy as np
x = np.array([1, 2, 3])              # vector, shape (3,)
A = np.array([[1,0,0], [0,1,1]])     # 2x3 matrix
A_ = A.T                             # transposed 3x2 matrix
```

### 8.1 Element-wise operations

```python
x + 1     # [2 3 4]
x * 2     # [2 4 6]
x ** 2    # [1 4 9]
x > 1     # [False  True  True]   ← boolean mask
x == 1    # [ True False False]
```

**`*` on arrays is element-wise, not matrix multiplication.**

### 8.2 Dot product; matrix-vector product

Three equivalent spellings each:

```python
x @ x         # 14        A @ x         # [1 5]
np.dot(x, x)  # 14        np.dot(A, x)  # [1 5]
x.dot(x)      # 14        A.dot(x)      # [1 5]
```

### 8.3 Summary statistics

```python
np.sum(x)    # 6
np.mean(x)   # 2.0
np.std(x)    # 0.816496580927726

np.sum(A,  axis=0)   # [1 1 1]         ← axis=0 collapses rows (column-wise result)
np.mean(A, axis=0)   # [0.5 0.5 0.5]
np.std(A,  axis=0)   # [0.5 0.5 0.5]
```

> **Mnemonic: `axis=k` is the axis that *disappears* from the output shape.** `A` is (2,3); `np.sum(A, axis=0)` is (3,).

### 8.4 Exercise 3

Setup:
```python
rng = np.random.default_rng(334)
x1 = rng.random(4)                    # shape (4,)
x2 = rng.integers(0, 20, size=(5,4))  # shape (5,4)
x3 = rng.random((5,4,3))              # shape (5,4,3)
x4 = rng.integers(0, 10, size=(5,4))  # shape (5,4)
```

> **Q3.1 — `.shape` vs `.size`.** `.shape` is the tuple of dimensions; `.size` is the **total number of elements** (the product of the shape). So `x3.shape == (5,4,3)` but `x3.size == 60`. Also `x3.ndim == 3`, and `len(x3)` gives only the **first** dimension, `5`.

> **Q3.2 — Multiplication.** `x2 * x4` is element-wise; both are (5,4), so the result is (5,4).
> `x2 @ x4` raises `ValueError` — the inner dimensions of (5,4)·(5,4) don't line up. You'd need `x2 @ x4.T` → (5,5) or `x2.T @ x4` → (4,4).
> Broadcasting also makes `x2 * x1` work — (5,4) and (4,) align on the trailing axis, giving (5,4).

> **Q3.3 — dtype gotcha.**
> ```python
> arr = np.array([1,2,3])   # dtype inferred as int64
> arr[0] = 1.5              # trying to change 1 to 1.5
> print(arr)                # [1 2 3]  ← silently truncated!
> ```
> NumPy arrays are **homogeneously typed with a fixed dtype**. Assigning a float into an int array truncates toward zero **with no warning**. Fix by creating it as float from the start — `np.array([1,2,3], dtype=float)` or `np.array([1.,2.,3.])` — or `arr = arr.astype(float)` first. A common silent-bug source in ML code.

---

## 9. Pandas

A **DataFrame** is a table just like a 2D NumPy array — but **rows/columns have names** and the table **can hold non-numbers**.

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

### 9.1 Slicing on a logical condition

```python
df[df['parity'] == 'odd']   # rows 0 and 2
```

### 9.2 Chaining the slice operator with summary statistics

```python
df[df['parity'] == 'odd']['x'].mean()   # 2.0
```

Read the pattern as: **build a boolean mask → filter rows → select a column → aggregate.** This idiom carries straight into Exercise 4 and HW1 Q4.

---

## 10. Matplotlib

Package for plotting data.

```python
import matplotlib.pylab as plt

x = np.arange(5)
y = x * x
plt.plot(x, y)
plt.show()
```

**Styling** — the format string is `color + marker + linestyle`:

```python
x = np.linspace(0, 1, 10)
plt.plot(x, x**2,      'ro:',  label="parabola")   # red, circles, dotted
plt.plot(x, np.sin(x), 'gx--', label="sine")       # green, x-marks, dashed
plt.xlabel('some numbers')
plt.ylabel('some more numbers')
plt.legend()
plt.show()
```

**Scatter:**

```python
x = np.random.uniform(1, 1000, 1000)
y = np.log(x) + np.random.normal(0, .3, 1000)
plt.scatter(x, y, s=5)   # s = marker size
plt.show()
```

---

## 11. Exercise 4 — Data Exploration (iris)

Download `iris_cs334.csv` from Canvas (In-Class Exercises folder) and upload it to Colab with the **upload button** in the file picker on the left sidebar.

> Colab's filesystem is **ephemeral** — uploaded files vanish when the runtime disconnects. Re-upload after a reconnect.

```python
df = pd.read_csv('iris_cs334.csv')
df   # 100 rows x 5 columns
```

Columns: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `variety`.
Varieties are **Setosa** and **Virginica** only — this trimmed version has 2 of the usual 3 classes, 50 rows each.

**Q4.1 Summary statistics** — the chaining idiom from §9.2:
```python
df['petal_length'].median()
df[df['variety'] == 'Virginica']['sepal_width'].mean()
df[df['variety'] == 'Setosa']['sepal_length'].std()
```

**Q4.2 Scatterplot** — `petal_length` vs `petal_width`, colored by variety. The two classes separate cleanly into two well-isolated blobs.

**Q4.3 Boxplot** — `sepal_length` grouped by variety. Virginica sits noticeably higher than Setosa, but with more overlap than the petal features show.

> **Why this matters beyond the exercise:** "these two classes form two separated blobs" is exactly the picture Lecture 03 opens with, and "can I draw a line between them?" is the whole of linear classification. The petal features are nearly separable; the sepal features are not. → `Lecture-03-Linear-Classification.md`

Worked answers and code: `exercises/ex02-python-numpy-pandas/`.

---

## Quick Reference

```
DIVISION        17 / 3  → 5.666…  true division, ALWAYS float
                17 // 3 → 5       floor division
                11.0 // 2.5 → 4.0  ← float in, float out
                17 % 3  → 2       remainder

ALIASING        b = a           two names, ONE object — never a copy
                b = a[:]        shallow copy (list)
                b = a.copy()    shallow copy (list or dict; a[:] fails on dict)
                copy.deepcopy(a)  nested structures

STRINGS         'CS' + 334  → TypeError, no implicit coercion
                'CS' + str(334)  or  f'{s}{x}'      → 'CS334'
                'CS' * 3    → 'CSCSCS'              repetition

SCOPE           indentation IS syntax — it defines the block
                return a, b  → comes back as a tuple
                locals are invisible outside the function

NUMPY           x * y   element-wise        x @ y   matrix / dot product
                arr = np.array([1,2,3]); arr[0] = 1.5  → [1 2 3]  SILENT truncation
                fixed dtype: np.array([1,2,3], dtype=float) or .astype(float)
                axis=k collapses axis k:  axis=0 → column-wise, axis=1 → row-wise
                .shape tuple of dims  |  .size total elements  |  .ndim rank
                len(arr) is only the FIRST dimension
                broadcasting aligns on TRAILING axes: (5,4) * (4,) → (5,4)

PANDAS          df[df[col] == val][other_col].agg()
                mask → filter rows → select column → aggregate

MATPLOTLIB      plt.plot(x, y, 'ro:')    fmt = color + marker + linestyle
                plt.scatter(x, y, s=5)   s = marker size
                plt.xlabel / ylabel / legend / show
```
