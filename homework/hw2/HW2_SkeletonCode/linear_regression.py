"""
Linear Regression
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data
import time

def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector 
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (N, 1). Each row is one instance.
        M: a non-negative integer
    
    Returns:
        Phi: np.array, shape (N, M+1)
    """
    N = X.shape[0]
    Phi = np.ones((N, M + 1))
    # column j is x^j (column 0 stays all 1s)
    for j in range(1, M + 1):
        Phi[:, j] = X[:, 0] ** j
    return Phi

def calculate_squared_loss(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)
    
    Returns:
        loss: float. The empirical risk based on squared loss as defined in the assignment.
    """
    N = len(y)
    predictions = np.dot(X, theta)
    errors = y - predictions
    loss = np.sum(errors ** 2) / (2 * N)
    return loss

def calculate_RMS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    N = len(y)
    predictions = np.dot(X, theta)
    errors = y - predictions
    E_rms = np.sqrt(np.sum(errors ** 2) / N)
    return E_rms


def ls_gradient_descent(X, y, learning_rate=0, return_n_iter=False):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float, the learning rate for GD
    
    Returns:
        theta: np.array, shape (d,)
        n_iter: int, number of theta updates (only if return_n_iter=True)
    """
    N = X.shape[0]
    d = X.shape[1]
    theta = np.zeros(d)
    prev_loss = calculate_squared_loss(X, y, theta)
    n_iter = 0

    while n_iter < 1000000:
        # one update per epoch, using all the points
        predictions = np.dot(X, theta)
        gradient = np.dot(X.T, predictions - y) / N
        theta = theta - learning_rate * gradient
        n_iter = n_iter + 1

        new_loss = calculate_squared_loss(X, y, theta)
        if abs(new_loss - prev_loss) <= 1e-10:
            break
        prev_loss = new_loss

    if return_n_iter:
        return theta, n_iter
    return theta


def ls_stochastic_gradient_descent(X, y, learning_rate=0, return_n_iter=False):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float or 'adaptive', the learning rate for SGD
    
    Returns:
        theta: np.array, shape (d,)
        n_iter: int, number of theta updates (only if return_n_iter=True)
    """
    N = X.shape[0]
    d = X.shape[1]
    theta = np.zeros(d)
    prev_loss = calculate_squared_loss(X, y, theta)
    n_iter = 0

    while n_iter < 1000000:
        # one epoch: update on each point, in order
        for i in range(N):
            if learning_rate == 'adaptive':
                # big steps at first, smaller steps later
                eta = 0.2 / (1 + n_iter / 100)
            else:
                eta = learning_rate
            error = y[i] - np.dot(X[i], theta)
            theta = theta + eta * error * X[i]
            n_iter = n_iter + 1
            if n_iter >= 1000000:
                break

        # check if we converged after each epoch
        new_loss = calculate_squared_loss(X, y, theta)
        if abs(new_loss - prev_loss) <= 1e-10:
            break
        prev_loss = new_loss

    if return_n_iter:
        return theta, n_iter
    return theta


def ls_closed_form_solution(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    d = X.shape[1]
    # theta = (X^T X + lambda I)^-1 X^T y
    A = np.dot(X.T, X) + reg_param * np.eye(d)
    theta = np.dot(np.dot(np.linalg.pinv(A), X.T), y)
    return theta


def weighted_ls_closed_form_solution(X, y, weights, reg_param=0):
    """
    Implements the closed form solution for weighted least squares regression.

    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        weights: np.array, shape (N,), the weights for each data point
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    d = X.shape[1]
    W = np.diag(weights)
    # theta = (X^T W X + lambda I)^-1 X^T W y
    A = np.dot(np.dot(X.T, W), X) + reg_param * np.eye(d)
    theta = np.dot(np.dot(np.dot(np.linalg.pinv(A), X.T), W), y)
    return theta


def part_1(fname_train):
    """
    This function should contain all the code you implement to complete part 1
    """
    print("========== Part 1 ==========")

    X_train, y_train = load_data(fname_train)
    Phi_train = generate_polynomial_features(X_train, 1)

    print("algorithm, eta, theta_0, theta_1, # iterations, runtime")

    for eta in [1e-4, 1e-3, 1e-2, 1e-1]:
        start = time.process_time()
        theta, n_iter = ls_gradient_descent(Phi_train, y_train, eta, return_n_iter=True)
        runtime = time.process_time() - start
        print("GD", eta, theta[0], theta[1], n_iter, runtime)

    for eta in [1e-4, 1e-3, 1e-2, 1e-1, 'adaptive']:
        start = time.process_time()
        theta, n_iter = ls_stochastic_gradient_descent(Phi_train, y_train, eta, return_n_iter=True)
        runtime = time.process_time() - start
        print("SGD", eta, theta[0], theta[1], n_iter, runtime)

    start = time.process_time()
    theta = ls_closed_form_solution(Phi_train, y_train)
    runtime = time.process_time() - start
    print("Closed form", theta[0], theta[1], runtime)

    print("Done!")


def part_2(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete part 2
    """
    print("=========== Part 2 ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    # (b) try M = 0 to 10
    Ms = list(range(11))
    train_errors = []
    val_errors = []
    for M in Ms:
        Phi_train = generate_polynomial_features(X_train, M)
        Phi_validation = generate_polynomial_features(X_validation, M)
        theta = ls_closed_form_solution(Phi_train, y_train)
        train_errors.append(calculate_RMS_Error(Phi_train, y_train, theta))
        val_errors.append(calculate_RMS_Error(Phi_validation, y_validation, theta))
        print("M =", M, "train:", train_errors[-1], "validation:", val_errors[-1])

    plt.figure()
    plt.plot(Ms, train_errors, "o-", color="red", label="Train")
    plt.plot(Ms, val_errors, "o-", color="blue", label="Validation")
    plt.xticks(Ms)
    plt.xlabel("M")
    plt.ylabel("RMS error")
    plt.legend()
    plt.savefig("../figures/rms_vs_M.png")
    plt.close()

    # (e) M = 10 with different lambdas
    lambdas = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    Phi_train = generate_polynomial_features(X_train, 10)
    Phi_validation = generate_polynomial_features(X_validation, 10)
    train_errors = []
    val_errors = []
    for lam in lambdas:
        theta = ls_closed_form_solution(Phi_train, y_train, reg_param=lam)
        train_errors.append(calculate_RMS_Error(Phi_train, y_train, theta))
        val_errors.append(calculate_RMS_Error(Phi_validation, y_validation, theta))
        print("lambda =", lam, "train:", train_errors[-1], "validation:", val_errors[-1])

    # can't put 0 on a log scale, so just space the points out evenly
    positions = list(range(len(lambdas)))
    plt.figure()
    plt.plot(positions, train_errors, "o-", color="red", label="Train")
    plt.plot(positions, val_errors, "o-", color="blue", label="Validation")
    plt.xticks(positions, ["0", "1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1"])
    plt.xlabel("lambda")
    plt.ylabel("RMS error")
    plt.legend()
    plt.savefig("../figures/rms_vs_lambda.png")
    plt.close()

    print("Done!")


def weighted_RMS_Error(X, y, theta, weights):
    errors = y - np.dot(X, theta)
    return np.sqrt(np.sum(weights * errors ** 2) / np.sum(weights))


def extra_credit(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete extra credit
    """
    print("=========== Extra Credit ==========")

    X_train, y_train, weights_train = load_data(fname_train, weighted=True)
    X_validation, y_validation, weights_validation = load_data(fname_validation, weighted=True)

    # same as part 2 (b) but with the weighted fit
    Ms = list(range(11))
    train_errors = []
    val_errors = []
    for M in Ms:
        Phi_train = generate_polynomial_features(X_train, M)
        Phi_validation = generate_polynomial_features(X_validation, M)
        theta = weighted_ls_closed_form_solution(Phi_train, y_train, weights_train)
        train_errors.append(weighted_RMS_Error(Phi_train, y_train, theta, weights_train))
        val_errors.append(weighted_RMS_Error(Phi_validation, y_validation, theta, weights_validation))
        print("M =", M, "weighted train:", train_errors[-1], "weighted validation:", val_errors[-1],
              "unweighted train:", calculate_RMS_Error(Phi_train, y_train, theta),
              "unweighted validation:", calculate_RMS_Error(Phi_validation, y_validation, theta))

    plt.figure()
    plt.plot(Ms, train_errors, "o-", color="red", label="Train")
    plt.plot(Ms, val_errors, "o-", color="blue", label="Validation")
    plt.xticks(Ms)
    plt.xlabel("M")
    plt.ylabel("weighted RMS error")
    plt.legend()
    plt.savefig("../figures/weighted_rms_vs_M.png")
    plt.close()

    # plot the weighted M=2 fit and the normal M=4 fit on top of the training data
    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    Phi_train_2 = generate_polynomial_features(X_train, 2)
    theta_weighted = weighted_ls_closed_form_solution(Phi_train_2, y_train, weights_train)
    Phi_train_4 = generate_polynomial_features(X_train, 4)
    theta_normal = ls_closed_form_solution(Phi_train_4, y_train)

    heavy = weights_train > 0.5
    light = weights_train < 0.5
    plt.figure()
    plt.scatter(X_train[heavy, 0], y_train[heavy], color="red", label="train, high weight")
    plt.scatter(X_train[light, 0], y_train[light], color="blue", marker="x", label="train, low weight")
    plt.plot(x_plot, np.dot(generate_polynomial_features(x_plot, 2), theta_weighted), "k--", label="weighted, M=2")
    plt.plot(x_plot, np.dot(generate_polynomial_features(x_plot, 4), theta_normal), color="green", label="unweighted, M=4")
    plt.ylim(-1.5, 1.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("../figures/weighted_fits.png")
    plt.close()

    print("Done!")


def main(fname_train, fname_validation):
    part_1(fname_train)
    part_2(fname_train, fname_validation)
    extra_credit(fname_train, fname_validation)


if __name__ == '__main__':
    main("data/linreg_train.csv", "data/linreg_validation.csv")
