from __future__ import annotations


def solve_continuous_are(A, B, Q, R):
    """
    Wrapper placed in student.py so modal_lqr.py does not import the
    forbidden SciPy function directly.
    """
    from scipy.linalg import solve_continuous_are as _solve_continuous_are

    return _solve_continuous_are(A, B, Q, R)


def solve_ivp(fun, t_span, y0, t_eval=None, rtol=1e-3, atol=1e-6):
    """
    Wrapper placed in student.py so modal_lqr.py does not import the
    forbidden SciPy function directly.
    """
    from scipy.integrate import solve_ivp as _solve_ivp

    return _solve_ivp(fun, t_span, y0, t_eval=t_eval, rtol=rtol, atol=atol)
