import numpy as np
import sympy as sp


def target_distribution(expression=None):
    """
    Create a target distribution function from a mathematical expression.

    Args:
        expression (str, optional): Mathematical expression as a string representing
            the target distribution. Should be a valid mathematical expression using
            'x' as the variable. Defaults to standard normal distribution if None.

    Returns:
        Callable[[float], float]: A function that takes a float value and returns
            the probability density at that point.

    Example:
        >>> # Create standard normal distribution
        >>> target_dist = target_distribution()
        >>> # Create custom distribution
        >>> target_dist = target_distribution('exp(-0.5 * (x - 2)**2) / sqrt(2 * pi)')
    """
    if expression is None:
        # Default to standard normal distribution
        expression = "exp(-0.5 * x**2) / sqrt(2 * pi)"

    try:
        # Attempt to parse the expression
        sympy_expr = sp.sympify(expression)
        
        # Check if 'x' is in the expression
        if 'x' not in str(sympy_expr.free_symbols):
            raise ValueError("Expression must contain the variable 'x'")
        
        # Convert to numpy function
        np_func = sp.lambdify('x', sympy_expr, modules=['numpy'])
        
        # Test evaluation
        try:
            test_value = float(np_func(0.0))
            if not np.isfinite(test_value):
                raise ValueError("Expression evaluates to non-finite value")
        except Exception as e:
            raise ValueError(f"Cannot evaluate expression: {str(e)}")
        
        return np_func
        
    except sp.SympifyError as e:
        raise ValueError(f"Cannot parse mathematical expression: {str(e)}")
    except ValueError as e:
        raise e  # Re-raise ValueErrors
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")


def proposal_distribution(x, variance=1.0):
    # Example proposal distribution: normal distribution centered at x
    return np.random.normal(x, np.sqrt(variance))
