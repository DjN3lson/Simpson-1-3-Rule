import math
import mpmath as mp
from calculus import integrate_simpson, _parse_expression, _max_abs_derivative, _composite_simpson

# ─────────────────────────── Helpers de input ───────────────────────────────

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
            print(f"  ⚠  '{raw}' no es un número válido.")

# ─────────────────────────── Tabla de convergencia ──────────────────────────
N_Values = [2, 4, 6, 8, 16, 32, 64, 128]

def run_convergence(function_text: str, a: float, b:float) -> None:
    x_sym, expr, fn = _parse_expression(function_text)

    #Referencia unica de alta precision
    reference = float(mp.quad(fn,[a,b]))

    # Cota: max|f''''| (se calcula una sola vez, es la misma para todos los n)
    max_d4 = _max_abs_derivative(x_sym, expr, 4, a, b)
    ba = b-a
    
    print()
    print("=" * 90)
    print(f"  Función  : {function_text}")
    print(f"  Intervalo: [{a}, {b}]")
    print(f"  Referencia (mp.quad): {reference:.10f}")
    if max_d4 is not None:
        print(f"  max|f''''| estimado : {max_d4:.6g}")
    print("=" * 90)

    # Encabezado de tabla
    print(f"\n  {'n':>5}  {'h':>12}  {'Simpson':>16}  {'mp.quad':>16}  "
          f"{'|Error abs|':>14}  {'Error rel %':>12}  {'Cota Lagrange':>15}")
    print("  " + "-" * 98)

    prev_error = None

    for n in N_Values:
        h = ba/n
        simpson_val, _= _composite_simpson(fn, a, b, n)
        abs_error = abs(reference - simpson_val)

        rel_error_str = (
            f"{(abs_error / abs(reference))*100:.6e}"
            if abs(reference) > 1e-300
            else "N/A"
        )


        if max_d4 is not None:
            bound = (ba ** 5 / (180.0 * n ** 4)) *max_d4
            bound_str = f"{bound:.6e}"
        else:
            bound_str = "N/A"

# Factor de reduccion del error respecto al n anterior
        ratio_str = ""
        if prev_error is not None and abs_error > 0:
            ratio = prev_error / abs_error
            ratio_str = f"  (x{ratio:.1f})"
        prev_error = abs_error

        print(f"  {n:>5}  {h:>12.6g}  {simpson_val:>16.10f}  {reference:>16.10f}  "
              f"{abs_error:>14.6e}  {rel_error_str:>12}  {bound_str:>15}{ratio_str}")

    print("  " + "-" * 98)
    print()
    print("  Nota: (xN) indica cuántas veces se redujo el error al duplicar n.")
    print("        Para Simpson 1/3 se espera una reducción aproximada de x16 (orden h^4).")
    print()

# ────────────────────────────── Main ────────────────────────────────────────

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║     ANÁLISIS DE CONVERGENCIA — Simpson 1/3 vs mpmath ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    function_text = ask("f(x)", "sin(x)")
    a = ask_float("Límite inferior a", 0.0)
    b = ask_float("Límite superior b", 3.1416)

    run_convergence(function_text, a, b)

if __name__ == "__main__":
    main()