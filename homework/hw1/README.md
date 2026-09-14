# HW1 — Issued Thu 08/27, Due **Sun 09/13, 11:59pm**

Spec: [`HW1-spec.pdf`](HW1-spec.pdf)

## Contents

| File | Role |
|---|---|
| `HW1-spec.pdf` | assignment spec |
| `HW1-written.pdf` | **HW1-Written submission** — 8 pages, Q1, Q2, Q3(f), Q5(a)–(d) |
| `HW1-written.tex` | source for the written PDF (`pdflatex HW1-written.tex`, run twice) |
| `figures/` | the four plots embedded in the written PDF, plus the raw Q3(f) timing table |
| `sumsquare.py` | **Q3** — vectorization comparison (5 functions) |
| `palmer.py` | **Q4** — pandas manipulation of the Palmer Penguins dataset (4 functions) |
| `README.txt` | signed honor-code statement — part of the code submission |
| `data/penguins.csv` | dataset for Q4/Q5 — **not** part of the submission |
| `test_hw1.py` | local self-check harness — **not** part of the submission |

## Status — ready to submit

| Question | Where | State |
|---|---|---|
| Q1, Q2 | `HW1-written.pdf` | done |
| Q3(a)–(e) | `sumsquare.py` | done — 9/9 harness checks |
| Q3(f) | `HW1-written.pdf` | done — timing table, log–log plot, discussion |
| Q4 | `palmer.py` | done — 7/7 harness checks, incl. the generality check |
| Q5(a)–(d) | `HW1-written.pdf` | done — plots, tables, Simpson's paradox, rules @ 96.49% |

## Submitting

Two separate Gradescope entries:

- **HW1-Written** — upload `HW1-written.pdf`. Pages must be **tagged to the right
  questions** on Gradescope.
- **HW1-Code** — upload **only** `sumsquare.py`, `palmer.py`, and `README.txt`.
  No data files, no `figures/`, no `test_hw1.py`. Re-uploads must include *all*
  files; only the latest submission counts.

## Running the self-check harness

```bash
cd homework/hw1
python3 test_hw1.py               # 16 checks
HW1_TRACE=1 python3 test_hw1.py   # with full tracebacks
```

The harness reads `data/penguins.csv` and also re-runs the Q4 chain against
`../../exercises/ex02-python-numpy-pandas/iris_cs334.csv` to confirm the functions
are **general**, not hardcoded to the penguins columns.

### Verified dataset facts (for sanity-checking output)

- `penguins.csv` is **344 × 8**
- `species` counts: Adelie **152**, Gentoo **124**, Chinstrap **68**
- **2 NaN** in each numeric column, **11 NaN** in `sex`

### The trap this harness exists to catch

`pd.get_dummies` returns **`bool`** columns, and `select_dtypes(include=[np.number])`
**does not match bool**. If `to_numeric` filters that way, the one-hot columns vanish
silently — you get 5 columns where you should have 8, with no error raised.
