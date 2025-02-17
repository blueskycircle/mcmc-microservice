import numpy as np
import sympy as sp


def target_distribution(expression=None):
    if expression is None:
        # Default to standard normal distribution
        expression = "exp(-0.5 * x**2) / sqrt(2 * pi)"

    # Convert the expression to a sympy expression
    sympy_expr = sp.sympify(expression)

    # Convert the sympy expression to a lambda function that uses numpy
    np_func = sp.lambdify(sp.symbols("x"), sympy_expr, modules="numpy")

    def distribution(x):
        return np_func(x)

    return distribution


def proposal_distribution(x):
    # Example proposal distribution: normal distribution centered at x
    return np.random.normal(x, 1.0)
