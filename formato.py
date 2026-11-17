"""Utilidades de formato para mostrar números con convención española (1.234,5)."""

import math


def numero(valor, decimales=1) -> str:
    """Formatea un número con punto de miles y coma decimal. Devuelve «—» si no hay valor."""
    if valor is None:
        return "—"
    try:
        if math.isnan(float(valor)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    texto = f"{float(valor):,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def porcentaje(valor, decimales=1) -> str:
    """Formatea una proporción 0-1 como porcentaje (0,265 -> «26,5 %»)."""
    if valor is None:
        return "—"
    return f"{numero(float(valor) * 100, decimales)} %"


def p_valor(p) -> str:
    """Formatea un valor p de forma legible (p < 0,001 cuando es muy pequeño)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 0.001:
        return "< 0,001"
    return numero(p, 3)
