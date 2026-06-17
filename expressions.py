"""
Safe math expression evaluator for step-dependent expressions.

Supports expressions with:
  - step (int): current 1-indexed step number
  - n / total_steps (int): total number of steps
  - Standard math functions via the math module
  - Standard Python operators: +, -, *, /, //, %, **, ==, !=, <, >, <=, >=, and, or, not

Examples:
  "step % 5 == 0"        -> True every 5th step
  "step >= 10 and step <= 20" -> True for steps 10-20
  "step == 1"             -> True for the first step
  "step == n"             -> True for the last step
  "sin(step / n * 3.14159)" -> sine ramp from 0 to pi
"""

import math
from typing import Any

# Allowed builtins: only safe ones
_ALLOWED_NAMES: dict[str, Any] = {
    # Math constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    # Math functions
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "bool": bool,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "pow": pow,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    "fmod": math.fmod,
    "fabs": math.fabs,
    "gcd": math.gcd,
    "degrees": math.degrees,
    "radians": math.radians,
    "clamp": lambda x, lo, hi: max(lo, min(x, hi)),
    "lerp": lambda a, b, t: a + (b - a) * t,
}


class StepContext:
    """Holds per-step context for expression evaluation."""

    def __init__(self, step: int, total_steps: int):
        self.step = step
        self.n = total_steps
        self.total_steps = total_steps

    def to_namespace(self) -> dict[str, Any]:
        return {"step": self.step, "n": self.n, "total_steps": self.total_steps}


def evaluate_expression(expression: str, step: int, total_steps: int) -> Any:
    """
    Evaluate a math expression in the context of a sampling step.

    Args:
        expression: The math expression string.
        step: Current step index (1-based).
        total_steps: Total number of steps.

    Returns:
        The evaluated result (bool, int, float, etc.)

    Raises:
        SyntaxError: If the expression has invalid syntax.
        ValueError: If the expression references disallowed names.
    """
    if not expression or not expression.strip():
        return True

    ctx = StepContext(step, total_steps)
    namespace = {**_ALLOWED_NAMES, **ctx.to_namespace()}

    # Compile for better error messages and slight speed improvement
    try:
        code = compile(expression.strip(), "<step_expression>", "eval")
    except SyntaxError:
        raise

    # Check that all names in the expression are allowed
    for name in code.co_names:
        if name not in namespace:
            raise ValueError(
                f"Expression uses disallowed name: '{name}'. "
                f"Allowed names: {sorted(namespace.keys())}"
            )

    return eval(code, {"__builtins__": {}}, namespace)


def evaluate_condition(expression: str, step: int, total_steps: int) -> bool:
    """
    Evaluate an expression and return its boolean truth value.

    This is the primary interface for step-filtering expressions.
    """
    result = evaluate_expression(expression, step, total_steps)
    return bool(result)


def evaluate_number(expression: str, step: int, total_steps: int) -> float:
    """
    Evaluate an expression and return it as a float.

    This is for expressions that compute a dynamic value (e.g., CFG schedule).
    """
    result = evaluate_expression(expression, step, total_steps)
    return float(result)
