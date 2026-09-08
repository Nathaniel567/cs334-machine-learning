"""
Local self-check harness for CS334 HW1 (Q3 sumsquare.py, Q4 palmer.py).

This file is a STUDY AID, not part of the graded submission.
Do NOT upload it to Gradescope -- HW1-Code accepts only:
    sumsquare.py, palmer.py, README.txt

Usage:
    python3 test_hw1.py

Every check prints PASS or FAIL with expected-vs-actual, so a failure tells you
which function broke and how. Unimplemented stubs simply show up as FAIL.
"""

import os
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PENGUINS = os.path.join(HERE, "data", "penguins.csv")
IRIS = os.path.join(HERE, "..", "..", "exercises",
                    "ex02-python-numpy-pandas", "iris_cs334.csv")

_results = []


def check(name, fn):
    """Run one check. fn returns (ok, detail) or raises."""
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, "raised {}: {}".format(type(e).__name__, e)
        if os.environ.get("HW1_TRACE"):
            traceback.print_exc()
    _results.append(ok)
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", name,
                               "" if ok else "\n         -> " + str(detail)))


def section(title):
    print("\n" + title)
    print("-" * len(title))


# ---------------------------------------------------------------- sumsquare.py

def test_sumsquare():
    section("Q3 -- sumsquare.py")
    try:
        import sumsquare as ss
    except Exception as e:
        print("  [FAIL] could not import sumsquare.py: {}".format(e))
        _results.append(False)
        return

    def t_gen_shape():
        x = ss.gen_random_samples(50)
        if not isinstance(x, np.ndarray):
            return False, "expected numpy.ndarray, got {}".format(type(x).__name__)
        return x.shape == (50,), "expected shape (50,), got {}".format(x.shape)

    def t_gen_dist():
        x = ss.gen_random_samples(200000)
        m, s = float(np.mean(x)), float(np.std(x))
        ok = abs(m) < 0.05 and abs(s - 1.0) < 0.05
        return ok, "expected mean~0 std~1, got mean={:.4f} std={:.4f}".format(m, s)

    def t_for_correct():
        x = np.array([1.0, 2.0, 3.0, 4.0])
        got, want = ss.sum_squares_for(x), 30.0
        return abs(float(got) - want) < 1e-9, "sum_squares_for({}) = {}, expected {}".format(
            list(x), got, want)

    def t_np_correct():
        x = np.array([1.0, 2.0, 3.0, 4.0])
        got, want = ss.sum_squares_np(x), 30.0
        return abs(float(got) - want) < 1e-9, "sum_squares_np({}) = {}, expected {}".format(
            list(x), got, want)

    def t_agree():
        x = ss.gen_random_samples(1000)
        a, b, want = ss.sum_squares_for(x), ss.sum_squares_np(x), float(np.sum(x ** 2))
        ok = abs(float(a) - want) < 1e-5 and abs(float(b) - want) < 1e-5
        return ok, "for={} np={} truth={}".format(a, b, want)

    def t_edge_single():
        x = np.array([3.0])
        a, b = ss.sum_squares_for(x), ss.sum_squares_np(x)
        ok = abs(float(a) - 9.0) < 1e-9 and abs(float(b) - 9.0) < 1e-9
        return ok, "single-element [3.0]: for={} np={}, expected 9.0".format(a, b)

    def t_edge_empty():
        x = np.array([])
        a, b = ss.sum_squares_for(x), ss.sum_squares_np(x)
        ok = abs(float(a)) < 1e-12 and abs(float(b)) < 1e-12
        return ok, "empty array: for={} np={}, expected 0".format(a, b)

    def t_time_ss():
        d = ss.time_ss([10, 100])
        if not isinstance(d, dict):
            return False, "expected dict, got {}".format(type(d).__name__)
        if set(d.keys()) != {"n", "ssfor", "ssnp"}:
            return False, "expected keys {{'n','ssfor','ssnp'}}, got {}".format(set(d.keys()))
        if list(d["n"]) != [10, 100]:
            return False, "d['n'] should echo input [10, 100], got {}".format(d["n"])
        if not (len(d["ssfor"]) == len(d["ssnp"]) == 2):
            return False, "ssfor/ssnp should each have 2 entries, got {} and {}".format(
                len(d["ssfor"]), len(d["ssnp"]))
        bad = [v for v in list(d["ssfor"]) + list(d["ssnp"])
               if not isinstance(v, float) or v < 0]
        return not bad, "all timings must be non-negative floats; offenders: {}".format(bad)

    def t_to_df():
        d = ss.time_ss([10, 100])
        df = ss.timess_to_df(d)
        if not isinstance(df, pd.DataFrame):
            return False, "expected pandas.DataFrame, got {}".format(type(df).__name__)
        cols = list(df.columns)
        if cols != ["n", "ssfor", "ssnp"]:
            return False, "columns must be exactly ['n','ssfor','ssnp'] IN ORDER, got {}".format(cols)
        return len(df) == 2, "expected 2 rows, got {}".format(len(df))

    check("gen_random_samples returns 1-D ndarray of right shape", t_gen_shape)
    check("gen_random_samples is ~Normal(0,1)", t_gen_dist)
    check("sum_squares_for correct on known input", t_for_correct)
    check("sum_squares_np correct on known input", t_np_correct)
    check("for/np agree with np.sum(x**2) to 5 decimals", t_agree)
    check("handles single-element array", t_edge_single)
    check("handles empty array", t_edge_empty)
    check("time_ss returns well-formed dict", t_time_ss)
    check("timess_to_df has exact column order n, ssfor, ssnp", t_to_df)


# ------------------------------------------------------------------- palmer.py

def test_palmer():
    section("Q4 -- palmer.py (Palmer Penguins)")
    try:
        import palmer as pm
    except Exception as e:
        print("  [FAIL] could not import palmer.py: {}".format(e))
        _results.append(False)
        return

    def t_load():
        df = pm.load_csv(PENGUINS)
        if not isinstance(df, pd.DataFrame):
            return False, "expected pandas.DataFrame, got {}".format(type(df).__name__)
        return df.shape == (344, 8), "expected shape (344, 8), got {}".format(df.shape)

    def t_remove_na_hits():
        df = pm.load_csv(PENGUINS)
        out = pm.remove_na(df, "bill_length_mm")
        if not isinstance(out, pd.DataFrame):
            return False, "expected DataFrame, got {}".format(type(out).__name__)
        n = len(out)
        if n != 342:
            return False, "expected 342 rows after dropping 2 NaN, got {}".format(n)
        left = int(out["bill_length_mm"].isna().sum())
        return left == 0, "expected 0 remaining NaN, got {}".format(left)

    def t_remove_na_none():
        # 'species' has zero NaN -- must still return a valid frame, not None/error
        df = pm.load_csv(PENGUINS)
        out = pm.remove_na(df, "species")
        if not isinstance(out, pd.DataFrame):
            return False, "expected DataFrame, got {}".format(type(out).__name__)
        return len(out) == 344, "no-NaN column should keep all 344 rows, got {}".format(len(out))

    def t_no_mutation():
        df = pm.load_csv(PENGUINS)
        before = df.shape
        pm.remove_na(df, "bill_length_mm")
        pm.onehot(df, "species")
        return df.shape == before, "input df was mutated: {} -> {}".format(before, df.shape)

    def t_onehot():
        df = pm.load_csv(PENGUINS)
        out = pm.onehot(df, "species")
        if not isinstance(out, pd.DataFrame):
            return False, "expected DataFrame, got {}".format(type(out).__name__)
        if "species" in out.columns:
            return False, "original column 'species' must be dropped; columns={}".format(
                list(out.columns))
        if len(out) != 344:
            return False, "row count should stay 344, got {}".format(len(out))
        if out.shape[1] != 10:
            return False, "expected 10 cols (8 - 1 + 3), got {}: {}".format(
                out.shape[1], list(out.columns))
        new = out.columns[-3:]
        sums = sorted(int(out[c].astype(int).sum()) for c in new)
        if sums != [68, 124, 152]:
            return False, ("last 3 columns should be the one-hot block summing to "
                           "[68,124,152] (Chinstrap/Gentoo/Adelie); got {} for {}".format(
                               sums, list(new)))
        for c in new:
            vals = set(np.unique(out[c].astype(int)))
            if not vals <= {0, 1}:
                return False, "column {} is not binary, values={}".format(c, vals)
        return True, ""

    def t_to_numeric():
        df = pm.load_csv(PENGUINS)
        df = pm.remove_na(df, "species")
        df = pm.onehot(df, "species")
        arr = pm.to_numeric(df)
        if not isinstance(arr, np.ndarray):
            return False, "expected numpy.ndarray, got {}".format(type(arr).__name__)
        if arr.dtype == object:
            return False, "array dtype is object -- non-numeric columns leaked through"
        if arr.shape[0] != 344:
            return False, "expected 344 rows, got {}".format(arr.shape[0])
        # 5 numeric originals (4 measurements + year) + 3 one-hot = 8.
        # Getting 5 here means the one-hot columns were bool and got dropped.
        if arr.shape[1] != 8:
            hint = ("  <-- looks like the one-hot columns were dropped; get_dummies "
                    "returns bool, which select_dtypes(np.number) does NOT match"
                    ) if arr.shape[1] == 5 else ""
            return False, "expected 8 columns, got {}{}".format(arr.shape[1], hint)
        return True, ""

    check("load_csv reads penguins into a 344x8 DataFrame", t_load)
    check("remove_na drops the 2 NaN rows (bill_length_mm -> 342)", t_remove_na_hits)
    check("remove_na handles a column with no NaN ('species')", t_remove_na_none)
    check("functions do not mutate the input DataFrame", t_no_mutation)
    check("onehot drops original col, appends 3 binary cols at the end", t_onehot)
    check("to_numeric returns ndarray and KEEPS the one-hot columns", t_to_numeric)


# --------------------------------------------------------- generality (not penguins)

def test_generality():
    section("Q4 -- generality check (must work for ANY csv, not just penguins)")
    if not os.path.exists(IRIS):
        print("  [SKIP] {} not found".format(os.path.normpath(IRIS)))
        return
    try:
        import palmer as pm
    except Exception as e:
        print("  [FAIL] could not import palmer.py: {}".format(e))
        _results.append(False)
        return

    def t_iris_chain():
        df = pm.load_csv(IRIS)
        if not isinstance(df, pd.DataFrame) or df.shape != (100, 5):
            return False, "load_csv(iris) expected (100, 5), got {}".format(
                getattr(df, "shape", type(df).__name__))
        df = pm.remove_na(df, "variety")
        if len(df) != 100:
            return False, "remove_na on no-NaN 'variety' expected 100 rows, got {}".format(len(df))
        df = pm.onehot(df, "variety")
        if "variety" in df.columns:
            return False, "'variety' should have been dropped by onehot"
        if df.shape[1] != 6:
            return False, "expected 6 cols (5 - 1 + 2), got {}: {}".format(
                df.shape[1], list(df.columns))
        arr = pm.to_numeric(df)
        if not isinstance(arr, np.ndarray):
            return False, "to_numeric expected ndarray, got {}".format(type(arr).__name__)
        if arr.shape != (100, 6):
            return False, "expected (100, 6) -- 4 measurements + 2 one-hot -- got {}".format(
                arr.shape)
        return True, ""

    check("same 4 functions work end-to-end on iris_cs334.csv", t_iris_chain)


if __name__ == "__main__":
    print("=" * 68)
    print("CS334 HW1 self-check  (study aid -- do NOT submit this file)")
    print("=" * 68)

    test_sumsquare()
    test_palmer()
    test_generality()

    total, passed = len(_results), sum(_results)
    print("\n" + "=" * 68)
    print("{} / {} checks passed".format(passed, total))
    if passed != total:
        print("Re-run with HW1_TRACE=1 to see full tracebacks for errors.")
    print("=" * 68)
