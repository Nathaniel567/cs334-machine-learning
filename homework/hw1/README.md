# HW1 — Issued Thu 08/27, Due **Sun 09/13, 11:59pm**

Spec: [`HW1-spec.pdf`](HW1-spec.pdf)

## Contents

| File | Role |
|---|---|
| `HW1-spec.pdf` | assignment spec |
| `sumsquare.py` | **Q3** — vectorization comparison (5 functions) |
| `palmer.py` | **Q4** — pandas manipulation of the Palmer Penguins dataset (4 functions) |
| `data/penguins.csv` | dataset for Q4/Q5 |
| `test_hw1.py` | local self-check harness — **not part of the submission** |

## Submitting

Two separate Gradescope entries:

- **HW1-Written** — one high-quality PDF, typed or handwritten. Pages must be **tagged to the right questions** on Gradescope. Code is not required in the PDF unless asked.
- **HW1-Code** — upload **only** `sumsquare.py`, `palmer.py`, and `README.txt`. **No data files.** Re-uploads must include *all* files; only the latest submission counts.

`README.txt` must contain the signed honor code statement from the spec:

```
THIS HOMEWORK IS MY OWN WORK, WRITTEN WITHOUT COPYING FROM OTHER STUDENTS
OR DIRECTLY FROM LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Any collaboration or external resources have been properly acknowledged.
<add details>
/* Your_Name_Here */
```

> `README.txt` is intentionally **not** in this repo — write it yourself when you submit, since it's a signed attestation.

## Running the self-check harness

```bash
cd homework/hw1
python3 test_hw1.py            # 16 checks
HW1_TRACE=1 python3 test_hw1.py   # with full tracebacks
```

The harness reads `data/penguins.csv` and also re-runs the Q4 chain against
`../../exercises/ex02-python-numpy-pandas/iris_cs334.csv` to confirm the functions
are **general**, not hardcoded to the penguins columns.

### Verified dataset facts (for sanity-checking your own output)

- `penguins.csv` is **344 × 8**
- `species` counts: Adelie **152**, Gentoo **124**, Chinstrap **68**
- **2 NaN** in each numeric column, **11 NaN** in `sex`

### The trap this harness exists to catch

`pd.get_dummies` returns **`bool`** columns, and `select_dtypes(include=[np.number])`
**does not match bool**. If `to_numeric` filters that way, the one-hot columns vanish
silently — you get 5 columns where you should have 8, with no error raised.

## Status

Q1/Q2 (written math) are handled separately. Q3/Q4/Q5 are the coding portion.

| Question | File | State |
|---|---|---|
| Q3 | `sumsquare.py` | not started — 5 functions, then the (f) timing sweep 10 → 10M and a log-log plot |
| Q4 | `palmer.py` | not started — `load_csv`, `remove_na`, `onehot`, `to_numeric` |
| Q5 | `viz.py` (to create) | not started — 2 bar plots, 4 boxplots, 6 scatterplots, then the (d) rules |

`viz.py`, `test_hw1.py`, and `data/` must **not** be uploaded to HW1-Code.
