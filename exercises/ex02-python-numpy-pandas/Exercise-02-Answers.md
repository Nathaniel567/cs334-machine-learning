# Exercise 02: Python — Answers

## Q1 Automatic Credit
Yes

---

## Q2 Python Basics

### Q2.1 Float Division
`11.0 // 2.5` returns:

```
4.0
```

### Q2.2 Adding Strings and Numbers
```python
s = 'CS'
x = 334
```
Correct ways to "add" `s` and `x` (all three evaluate to `'CS334'`):
- `s + str(x)`
- `'{}{}'.format(s, x)`
- `f'{s}{x}'`

Not correct:
- `s + x` → `TypeError: can only concatenate str (not "int") to str`
- `s + chr(x)` → `'CSŎ'` (`chr(334)` is a unicode character, not the digits `334`)
- `s + hex(x)` → `'CS0x14e'` (hex representation, not decimal)
- `hash(s) + x` → an integer (e.g. `-6389359129748080843`), not a string at all

### Q2.3 Copying Dictionaries
```python
dct1 = {'hello': 'world!', "it's!": 'me!'}
dct2 = dct1  ### <- fix this
dct2["it's!"] = 'you!'
```

**Fix:** `dct2 = dct1.copy()`

**Why:** `dct2 = dct1` just binds a second name to the *same* dictionary object (aliasing), so mutating `dct2` also mutates `dct1`. `dct1.copy()` (or `dict(dct1)`) creates a separate dictionary object, so changing `dct2` no longer changes `dct1`.

---

## Q3 NumPy Basics

Setup (must be run in this exact order):
```python
import numpy as np
rng = np.random.default_rng(334)
x1 = rng.random(4)
x2 = rng.integers(0, 20, size=(5,4))
x3 = rng.random((5,4,3))
x4 = rng.integers(0, 10, size=(5,4))
```

### Q3.1 shape vs. size
- `x3.shape` → `(5, 4, 3)`
- `x2.size` → `20`

### Q3.2 Multiplication
Element-wise product `x2 * x4`:
```
[[ 64  36   0  30]
 [ 15  44   6  95]
 [  3   1  60 136]
 [117  96  98  96]
 [ 21   4  30  56]]
```

Matrix-vector product `x2 @ x1`:
```
[ 7.88095035 11.84467672  7.03564031 15.16760187  9.69828305]
```

### Q3.3 numpy dtype
```python
import numpy as np
arr = np.array([1,2,3])
arr[0] = 1.5
print(arr)  # [1 2 3]
```

**Fix:** `arr = np.array([1,2,3], dtype=float)`

**Why:** `np.array([1,2,3])` infers an integer dtype (`int64`). NumPy arrays are homogeneously typed with a fixed dtype, so assigning a float (`1.5`) into an int array silently truncates it to `1` instead of raising an error or upcasting. Declaring the array as `dtype=float` (or calling `.astype(float)`) avoids the silent truncation.

---

## Q4 Data Loading + Visualization

Data loaded from `iris_cs334.csv` (this folder) with:
```python
import pandas as pd
df = pd.read_csv('iris_cs334.csv')
```

### Q4.1 Summary statistics
- Median of `petal_length` (all samples): **3.20**
- Mean of `sepal_width` for `Virginica`: **2.97**
- Std of `sepal_length` for `Setosa`: **0.35**

Code:
```python
df['petal_length'].median()                              # 3.2
df[df['variety'] == 'Virginica']['sepal_width'].mean()   # 2.974   -> 2.97
df[df['variety'] == 'Setosa']['sepal_length'].std()      # 0.35249 -> 0.35
```

### Q4.2 Scatterplot
Petal width (x-axis) vs. petal length (y-axis), colored by variety: `q4_2_scatterplot.png` (this folder).

```python
import matplotlib.pylab as plt

for variety, group in df.groupby('variety'):
    plt.scatter(group['petal_width'], group['petal_length'], label=variety, s=20)
plt.xlabel('petal_width')
plt.ylabel('petal_length')
plt.legend()
plt.savefig('q4_2_scatterplot.png', dpi=150, bbox_inches='tight')
```

### Q4.3 Boxplot
Sepal length (y-axis) grouped by variety (x-axis): `q4_3_boxplot.png` (this folder).

```python
varieties = sorted(df['variety'].unique())
data = [df[df['variety'] == v]['sepal_length'] for v in varieties]
plt.boxplot(data, labels=varieties)
plt.xlabel('variety')
plt.ylabel('sepal_length')
plt.savefig('q4_3_boxplot.png', dpi=150, bbox_inches='tight')
```

---

*All numeric answers computed and verified via `solve.py` in this folder, run against `iris_cs334.csv`.*
