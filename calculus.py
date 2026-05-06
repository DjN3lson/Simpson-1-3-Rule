import math
from dataclasses import dataclass

import mpmath as mp
import sympy as sp


# ─────────────────────────── Estructuras de datos ───────────────────────────

@dataclass
class WeightedNode: # Class to store the nodes of the method
    i: int
    xi: float
    fxi: float
    weight: float  # 1, 2 o 4


@dataclass
class MethodSteps: # Class to store the steps of the method
    rule_name: str
    lagrange_order: int
    lagrange_derivation: str
    formula_text: str
    h: float
    multiplier: float
    nodes: list
    weighted_sum: float
    result: float


@dataclass
class IntegrationResult: # Class to store the result of the integration
    method: str
    rule: str
    steps: MethodSteps
    numerical_result: float
    reference_integral: float
    absolute_error: float
    derivative_order_used: int | None
    derivative_max_abs: float | None
    lagrange_error_bound: float | None
    lagrange_check_passed: bool | None
    relative_error: float | None
    error_bound_relative: float | None
    error_bound_formula: str
    details: str


class FunctionParseError(ValueError):
    pass


# ──────────────────────────── Parseo de expresión ───────────────────────────

def _parse_expression(expression_text: str) -> tuple[sp.Symbol, sp.Expr, callable]:
    variable = sp.symbols("x")
    try:
        expr = sp.sympify(expression_text)
    except Exception as exc:
        raise FunctionParseError("No se pudo interpretar la funcion.") from exc

    if any(sym != variable for sym in expr.free_symbols):
        raise FunctionParseError("La expresion solo puede contener la variable x.")

    try:
        fn = sp.lambdify(variable, expr, modules=["mpmath", "math"])
        _ = fn(0.0)
    except Exception as exc:
        raise FunctionParseError("La funcion no se puede evaluar numericamente.") from exc

    return variable, expr, fn


# ─────────────────────── Simpson 1/3 compuesto ──────────────────────────────

def _composite_simpson(fn: callable, a: float, b: float, n: int) -> tuple[float, MethodSteps]:
    if n % 2 != 0:
        raise ValueError("Para Simpson 1/3 compuesto, n debe ser par.")
    if n < 2:
        raise ValueError("Para Simpson 1/3 compuesto, n debe ser >= 2.")

    h = (b - a) / n #para todas las particiones. Libro lo hace solo para cierto caso, h = x2-x1 = x1-x0
    nodes: list[WeightedNode] = []

    #1: primer y ultimo punto (nodo), 4: puntos impares, 2: puntos pares
    for i in range(n + 1):
        xi = a + i * h
        fxi = float(fn(xi))
        if i == 0 or i == n:
            weight = 1.0
        elif i % 2 == 1:
            weight = 4.0
        else:
            weight = 2.0
        nodes.append(WeightedNode(i=i, xi=xi, fxi=fxi, weight=weight)) 

    weighted_sum = sum(nd.weight * nd.fxi for nd in nodes) #Sumatoria de pesos * f(xi)
    result = (h / 3.0) * weighted_sum

    formula_text = (
        f"(h/3) * [f(x0) + 4*f(x1) + 2*f(x2) + ... + 4*f(x{n-1}) + f(x{n})]"
        f"  con  h = {h:.6g}"
    )

    steps = MethodSteps(
        rule_name="Simpson 1/3 compuesto",
        lagrange_order=2,
        lagrange_derivation=(
            "Se aproxima f(x) en cada panel doble con el polinomio interpolador\n"
            "de Lagrange de GRADO 2 a traves de tres nodos:\n"
            "  (x0, f(x0)),  (x1=x0+h, f(x1)),  (x2=x0+2h, f(x2))\n\n"
            "  P2(x) = f(x0)*L0(x) + f(x1)*L1(x) + f(x2)*L2(x)\n\n"
            "  Polinomios base de Lagrange:\n"
            "    L0(x) = (x-x1)(x-x2) / [(x0-x1)(x0-x2)]\n"
            "    L1(x) = (x-x0)(x-x2) / [(x1-x0)(x1-x2)]\n"
            "    L2(x) = (x-x0)(x-x1) / [(x2-x0)(x2-x1)]\n\n"
            "Integrando P2 de x0 a x2 (ancho total = 2h):\n"
            "  integral_panel = (h/3) * [f(x0) + 4*f(x1) + f(x2)]\n\n"
            "Aplicando a cada par de subintervalos (n par obligatorio):\n"
            "  integral compuesta = (h/3)*[f(x0)+4*f(x1)+2*f(x2)+...+4*f(xn-1)+f(xn)]\n\n"
            "Cota del error: |E| <= (b-a)^5 / (180*n^4) * max|f''''(c)|"
        ),
        formula_text=formula_text,
        h=h,
        multiplier=h / 3.0,
        nodes=nodes,
        weighted_sum=weighted_sum,
        result=result,
    )
    return result, steps


# ──────────────────────── Derivada máxima en [a,b] ──────────────────────────
#Estima el máximo de |f''''(x)| en [a, b], necesario para la cota de Lagrange:
def _max_abs_derivative(variable: sp.Symbol, expr: sp.Expr, order: int, a: float, b: float) -> float | None:
    derivative_expr = sp.diff(expr, variable, order)
    derivative_fn = sp.lambdify(variable, derivative_expr, modules=["mpmath", "math"])

    sample_count = 600
    max_abs_value = -1.0

    for i in range(sample_count + 1):
        xi = a + (b - a) * i / sample_count
        try:
            value = derivative_fn(xi)
            value_abs = abs(float(value))
            if math.isfinite(value_abs):
                max_abs_value = max(max_abs_value, value_abs)
        except Exception:
            continue

    return max_abs_value if max_abs_value >= 0 else None


# ──────────────────────────── API pública ───────────────────────────────────
#función que llama main.py. Orquesta todo el proceso
def integrate_simpson(
    function_text: str,
    a: float,
    b: float,
    n: int,
) -> IntegrationResult:
    """
    Integra f(x) en [a, b] con n subintervalos usando Simpson 1/3 compuesto.
    n debe ser par y >= 2.
    """
    x, expr, fn = _parse_expression(function_text)

    # Caso degenerado
    if a == b:
        fxa = float(fn(a))
        dummy_steps = MethodSteps(
            rule_name="Simpson 1/3 compuesto",
            lagrange_order=2,
            lagrange_derivation="",
            formula_text="",
            h=0.0,
            multiplier=0.0,
            nodes=[WeightedNode(i=0, xi=a, fxi=fxa, weight=1.0)],
            weighted_sum=fxa,
            result=0.0,
        )
        return IntegrationResult(
            method="direct",
            rule="simpson",
            steps=dummy_steps,
            numerical_result=0.0,
            reference_integral=0.0,
            absolute_error=0.0,
            derivative_order_used=4,
            derivative_max_abs=0.0,
            lagrange_error_bound=0.0,
            lagrange_check_passed=True,
            relative_error=0.0,
            error_bound_relative=0.0,
            error_bound_formula="Intervalo degenerado: a == b.",
            details="Intervalo degenerado: a == b.",
        )

    if n < 2:
        raise ValueError("n debe ser >= 2 para Simpson 1/3.")

    numerical_value, method_steps = _composite_simpson(fn, a, b, n)
    derivative_order = 4

    reference = float(mp.quad(fn, [a, b])) #integral de referencia por la biblioteca mpmath
    abs_error = abs(reference - numerical_value) #error absoluto
    max_derivative = _max_abs_derivative(x, expr, derivative_order, a, b)

    #Error relativo
    rel_err: float | None = (
        (abs_error / abs(reference)) * 100.0
        if math.isfinite(reference) and abs(reference) > 1e-300
        else None
    )

    details = "Comprobacion de Lagrange con cota de Simpson 1/3 compuesto usando max|f''''|."

    if max_derivative is None:
        return IntegrationResult(
            method="direct",
            rule="simpson",
            steps=method_steps,
            numerical_result=numerical_value,
            reference_integral=reference,
            absolute_error=abs_error,
            derivative_order_used=derivative_order,
            derivative_max_abs=None,
            lagrange_error_bound=None,
            lagrange_check_passed=None,
            relative_error=rel_err,
            error_bound_relative=None,
            error_bound_formula="No fue posible estimar la derivada requerida.",
            details="No fue posible estimar la derivada requerida para la cota teorica del error.",
        )

    #Cota teórica del Lagrange. Calculada
    ba = b - a
    bound = (ba ** 5 / (180.0 * n ** 4)) * max_derivative
    formula = (
        f"|E_S1/3| <= (b-a)^5 / (180*n^4) * max|f''''|\n"
        f"       = ({ba:.6g})^5 / (180 * {n}^4) * {max_derivative:.6g}\n"
        f"       = {bound:.6g}"
    )

    #Cota relativa
    bound_rel: float | None = (
        (bound / abs(reference)) * 100.0
        if math.isfinite(reference) and abs(reference) > 1e-300
        else None
    )

    return IntegrationResult(
        method="direct",
        rule="simpson",
        steps=method_steps,
        numerical_result=numerical_value,
        reference_integral=reference,
        absolute_error=abs_error,
        derivative_order_used=derivative_order,
        derivative_max_abs=max_derivative,
        lagrange_error_bound=bound,
        lagrange_check_passed=abs_error <= bound + 1e-12,
        relative_error=rel_err,
        error_bound_relative=bound_rel,
        error_bound_formula=formula,
        details=details,
    )
