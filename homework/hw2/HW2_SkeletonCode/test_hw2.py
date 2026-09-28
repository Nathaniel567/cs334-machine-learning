"""
Local self-check harness for HW2 -- NOT part of the submission.
Run from HW2_SkeletonCode/:  python3 test_hw2.py   (HW2_TRACE=1 for tracebacks)
"""
import os
import time
import traceback

import numpy as np

import perceptron as pc
import linear_regression as lr
from helper import load_data

results = []


def check(name):
    def wrap(fn):
        try:
            fn()
            results.append((name, True, ""))
        except Exception as e:
            if os.environ.get("HW2_TRACE"):
                traceback.print_exc()
            results.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return wrap


X_cls, y_cls = load_data("data/classification.csv", d=2)
X_tr, y_tr = load_data("data/linreg_train.csv")
X_va, y_va = load_data("data/linreg_validation.csv")
Phi1 = lr.generate_polynomial_features(X_tr, 1)
theta_ref = np.polyfit(X_tr[:, 0], y_tr, 1)[::-1]


# ---------------- perceptron ----------------
@check("all_correct: boundary point counts as misclassified")
def _():
    X = np.array([[1.0, 0.0], [2.0, 0.0]])
    y = np.array([1, 1])
    assert not pc.all_correct(X, y, np.array([1.0, 0.0]), -1.0)   # first point on boundary
    assert pc.all_correct(X, y, np.array([1.0, 0.0]), -0.5)


@check("all_correct: returns a plain bool")
def _():
    assert type(pc.all_correct(X_cls, y_cls, np.zeros(2), 0.0)) is bool


@check("perceptron: separates classification.csv")
def _():
    theta, b, _ = pc.perceptron(X_cls, y_cls)
    assert np.all(y_cls * (X_cls @ theta + b) > 0)


@check("perceptron: theta = sum a_i y_i x_i, b = sum a_i y_i  (Q2a)")
def _():
    theta, b, alpha = pc.perceptron(X_cls, y_cls)
    assert alpha.shape == (len(y_cls),)
    assert np.allclose(theta, (alpha * y_cls) @ X_cls)
    assert np.isclose(b, np.sum(alpha * y_cls))


@check("perceptron: in-order pass gives known result on toy data")
def _():
    X = np.array([[1.0, 1.0], [-1.0, -1.0]])
    y = np.array([1, -1])
    theta, b, alpha = pc.perceptron(X, y)
    # zero init: point 0 on boundary -> update to theta=[1,1], b=1; point 1 then correct
    assert np.allclose(theta, [1, 1]) and b == 1 and list(alpha) == [1, 0]


# ---------------- features / losses ----------------
@check("generate_polynomial_features: shape and columns")
def _():
    Phi = lr.generate_polynomial_features(np.array([[2.0], [3.0]]), 3)
    assert np.allclose(Phi, [[1, 2, 4, 8], [1, 3, 9, 27]])
    assert lr.generate_polynomial_features(X_tr, 0).shape == (20, 1)


@check("calculate_squared_loss: includes the 1/2 factor")
def _():
    X = np.array([[1.0], [1.0]])
    assert np.isclose(lr.calculate_squared_loss(X, np.array([2.0, 0.0]), np.array([0.0])), 1.0)


@check("calculate_RMS_Error: no 1/2 factor")
def _():
    X = np.array([[1.0], [1.0]])
    assert np.isclose(lr.calculate_RMS_Error(X, np.array([2.0, 0.0]), np.array([0.0])), np.sqrt(2))


# ---------------- optimizers ----------------
@check("closed form matches np.polyfit (M=1)")
def _():
    assert np.allclose(lr.ls_closed_form_solution(Phi1, y_tr), theta_ref)


@check("closed form with reg_param matches ridge normal equations")
def _():
    Phi = lr.generate_polynomial_features(X_tr, 10)
    lam = 1e-3
    expected = np.linalg.solve(Phi.T @ Phi + lam * np.eye(11), Phi.T @ y_tr)
    assert np.allclose(lr.ls_closed_form_solution(Phi, y_tr, reg_param=lam), expected, atol=1e-6)


@check("GD eta=0.1 lands near closed form; default call returns ndarray only")
def _():
    theta = lr.ls_gradient_descent(Phi1, y_tr, learning_rate=0.1)
    assert isinstance(theta, np.ndarray) and theta.shape == (2,)
    assert np.allclose(theta, theta_ref, atol=1e-3)


@check("SGD iterations are a multiple of N (per-example updates, per-epoch check)")
def _():
    _, n = lr.ls_stochastic_gradient_descent(Phi1, y_tr, 1e-2, return_n_iter=True)
    assert n % len(y_tr) == 0


@check("SGD solves a 2-point, 2-parameter problem exactly")
def _():
    Phi = np.array([[1.0, 2.0], [1.0, -1.0]])
    y = np.array([1.0, 0.0])
    theta = lr.ls_stochastic_gradient_descent(Phi, y, 0.1)
    assert np.allclose(Phi @ theta, y, atol=1e-3)


@check("GD/SGD return_n_iter: GD counts epochs, SGD counts per-example updates")
def _():
    _, n_gd = lr.ls_gradient_descent(Phi1, y_tr, 1e-2, return_n_iter=True)
    _, n_sgd = lr.ls_stochastic_gradient_descent(Phi1, y_tr, 1e-2, return_n_iter=True)
    assert 0 < n_gd < 1e6 and 0 < n_sgd < 1e6


@check("SGD adaptive: runs, converges near closed form")
def _():
    theta = lr.ls_stochastic_gradient_descent(Phi1, y_tr, 'adaptive')
    assert np.allclose(theta, theta_ref, atol=1e-3)


@check("weighted closed form: uniform weights == ordinary LS; matches formula")
def _():
    Phi = lr.generate_polynomial_features(X_tr, 3)
    assert np.allclose(lr.weighted_ls_closed_form_solution(Phi, y_tr, np.ones(20)),
                       lr.ls_closed_form_solution(Phi, y_tr))
    _, _, w = load_data("data/linreg_train.csv", weighted=True)
    W = np.diag(w)
    expected = np.linalg.solve(Phi.T @ W @ Phi + 0.1 * np.eye(4), Phi.T @ W @ y_tr)
    assert np.allclose(lr.weighted_ls_closed_form_solution(Phi, y_tr, w, reg_param=0.1), expected)


@check("full linear_regression.main runs in < 60s")
def _():
    import contextlib, io
    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        lr.main("data/linreg_train.csv", "data/linreg_validation.csv")
    assert time.time() - start < 60


passed = sum(ok for _, ok, _ in results)
for name, ok, msg in results:
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {msg}"))
print(f"\n{passed}/{len(results)} checks passed")
