"""
Calculadora de Integración Numérica — CLI
Método: Simpson 1/3 compuesto
Con verificación de cota de error de Lagrange
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calculus import FunctionParseError, integrate_simpson

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de UI
# ──────────────────────────────────────────────────────────────────────────────

WIDTH = 70

def sep(char="─"):
    print(char * WIDTH)

def header(title: str):
    sep("═")
    print(f"  {title}")
    sep("═")

def section(title: str):
    print()
    sep()
    print(f"  {title}")
    sep()

def ask(prompt: str, default: str | None = None) -> str:
    hint = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"  {prompt}{hint}: ").strip()
        if value == "" and default is not None:
            return default
        if value:
            return value
        print("  ⚠  Este campo es obligatorio.")

def ask_float(prompt: str, default: float | None = None) -> float:
    default_str = str(default) if default is not None else None
    while True:
        raw = ask(prompt, default_str)
        try:
            return float(raw)
        except ValueError:
            print(f"  ⚠  '{raw}' no es un número válido. Intenta de nuevo.")

def ask_int(prompt: str, default: int | None = None, min_val: int = 2) -> int:
    default_str = str(default) if default is not None else None
    while True:
        raw = ask(prompt, default_str)
        try:
            val = int(raw)
            if val < min_val:
                print(f"  ⚠  El valor mínimo es {min_val}.")
                continue
            if val % 2 != 0:
                print("  ⚠  n debe ser par para Simpson 1/3.")
                continue
            return val
        except ValueError:
            print(f"  ⚠  '{raw}' no es un entero válido.")

def ask_bool(prompt: str, default: bool = False) -> bool:
    default_str = "s" if default else "n"
    while True:
        raw = ask(f"{prompt} (s/n)", default_str).lower()
        if raw in ("s", "si", "sí", "y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  ⚠  Responde 's' o 'n'.")

# ──────────────────────────────────────────────────────────────────────────────
# Recolección de inputs
# ──────────────────────────────────────────────────────────────────────────────

def collect_inputs() -> dict:
    header("CALCULADORA DE INTEGRACIÓN NUMÉRICA — SIMPSON 1/3")
    print()
    print("  Escribe la función en términos de x usando sintaxis Python.")
    print("  Ejemplos: sin(x) + x**2 | exp(-x**2) | 1/(1+x) | x*log(x)")
    print()

    function = ask("f(x)")
    a        = ask_float("Límite inferior a")
    b        = ask_float("Límite superior b")
    n        = ask_int("Número de subintervalos n (par, >= 2)", default=4, min_val=2)

    return dict(function=function, a=a, b=b, n=n)

# ──────────────────────────────────────────────────────────────────────────────
# Mostrar resultados
# ──────────────────────────────────────────────────────────────────────────────

def print_results(params: dict, res) -> None:
    s = res.steps

    # ── Configuración ─────────────────────────────────────────────────────────
    section("CONFIGURACIÓN")
    print(f"  Función      : f(x) = {params['function']}")
    print(f"  Intervalo    : [{params['a']}, {params['b']}]")
    print(f"  Subintervalos: n = {params['n']}")
    print(f"  Método       : Simpson 1/3 compuesto")

    # ── Derivación de Lagrange ────────────────────────────────────────────────
    section("DERIVACIÓN DEL MÉTODO (LAGRANGE)")
    for line in s.lagrange_derivation.splitlines():
        print(f"  {line}")

    # ── Tabla de nodos ────────────────────────────────────────────────────────
    section("TABLA DE NODOS")
    col_w = [6, 16, 16, 8]
    headers = ["  i", "      xi", "    f(xi)", " peso"]
    print("  " + "  ".join(h.ljust(col_w[j]) for j, h in enumerate(headers)))
    sep("-")
    for nd in s.nodes:
        print(
            f"  {nd.i:<{col_w[0]}}  "
            f"{nd.xi:>{col_w[1]}.8f}  "
            f"{nd.fxi:>{col_w[2]}.8f}  "
            f"{nd.weight:>{col_w[3]}g}"
        )

    # ── Cálculo paso a paso ───────────────────────────────────────────────────
    section("CÁLCULO NUMÉRICO")
    print(f"  h              = {s.h:.8g}")
    print(f"  Multiplicador  = {s.multiplier:.8g}")
    print(f"  Suma ponderada = {s.weighted_sum:.8g}")
    print(f"  Fórmula        : {s.formula_text}")
    print()
    print(f"  ┌─ RESULTADO NUMÉRICO ─────────────────────────────────────┐")
    print(f"  │  ∫ f(x) dx ≈ {res.numerical_result:.10g}".ljust(WIDTH - 2) + "│")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # ── Análisis de error ─────────────────────────────────────────────────────
    section("ANÁLISIS DE ERROR")
    print(f"  Integral de referencia (mpmath) : {res.reference_integral:.10g}")
    print(f"  Error absoluto                  : {res.absolute_error:.6e}")
    if res.relative_error is not None:
        print(f"  Error relativo                  : {res.relative_error:.4f} %")

    print()
    print("  Cota teórica de Lagrange:")
    for line in res.error_bound_formula.splitlines():
        print(f"    {line}")

    if res.lagrange_error_bound is not None:
        print()
        print(f"  Cota calculada : {res.lagrange_error_bound:.6e}")
        if res.error_bound_relative is not None:
            print(f"  Cota relativa  : {res.error_bound_relative:.4f} %")
        check_sym = "✔" if res.lagrange_check_passed else "✘"
        check_msg = "Se cumple" if res.lagrange_check_passed else "NO se cumple (revisar n)"
        print(f"  Verificación   : {check_sym}  {check_msg}")

    print()
    print(f"  {res.details}")
    sep("═")

# ──────────────────────────────────────────────────────────────────────────────
# Bucle principal
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.system("cls" if os.name == "nt" else "clear")
    while True:
        try:
            params = collect_inputs()
        except KeyboardInterrupt:
            print("\n\n  Saliendo... ¡hasta luego!\n")
            break

        print()
        print("  Calculando", end="", flush=True)

        try:
            result = integrate_simpson(
                function_text = params["function"],
                a             = params["a"],
                b             = params["b"],
                n             = params["n"],
            )
            print(" ✔")
            print_results(params, result)

        except FunctionParseError as exc:
            print(f"\n  ✘ Error al interpretar la función: {exc}")
        except ValueError as exc:
            print(f"\n  ✘ Error de parámetros: {exc}")
        except Exception as exc:
            print(f"\n  ✘ Error inesperado: {exc}")

        print()
        try:
            again = ask_bool("¿Calcular otra integral?", default=True)
        except KeyboardInterrupt:
            again = False

        if not again:
            print("\n  ¡Hasta luego!\n")
            break

        os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()