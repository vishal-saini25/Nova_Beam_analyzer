"""Thin, text-only wrapper around the supplied verified beam solver."""

from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import numpy as np

from beam_analyser import build_model, solve_beam, report


class BeamInputError(ValueError):
    pass


def analyse(config: dict[str, Any]) -> dict[str, Any]:
    """Validate, solve and return a deterministic text report (no plot files)."""
    if not isinstance(config, dict):
        raise BeamInputError("Beam configuration must be a JSON object.")

    required = ("L", "section", "supports")
    missing = [key for key in required if key not in config or config[key] is None]
    if missing:
        raise BeamInputError("Missing required configuration fields: " + ", ".join(missing))

    try:
        model = build_model(config)
        solution = solve_beam(model)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise BeamInputError(str(exc)) from exc

    output = StringIO()
    with redirect_stdout(output):
        report(solution)
    report_text = output.getvalue().strip()

    # A text-only substitute for plotted SFD/BMD: report ordinates at every
    # support, applied-load boundary and requested evaluation point.
    stations = {0.0, float(model.L)}
    stations.update(float(item.x) for item in model.supports)
    stations.update(float(item.a) for item in model.point_loads)
    stations.update(float(item.a) for item in model.couples)
    for item in model.dist_loads:
        stations.update((float(item.start), float(item.end)))
    stations.update(float(x) for x in model.query_points)
    report_lines = [
        "\nKEY POSITION ORDINATES (internal load boundaries show the solver right-hand value)",
        "x (m) | Shear V (kN) | Moment M (kN.m) | Slope (rad) | Deflection (mm)",
    ]
    for x in sorted(stations):
        values = solution.at(x)
        report_lines.append(
            f"{x:.4f} | {values['V']/1e3:.4f} | {values['M']/1e3:.4f} | "
            f"{values['theta']:.6e} | {values['y']*1e3:.6f}"
        )
    report_text += "\n\n" + "\n".join(report_lines)

    reaction_sum = sum(item["R"] for item in solution.reactions)
    total_load = solution.total_load()
    equilibrium_error = abs(reaction_sum - total_load)
    equilibrium_ok = bool(equilibrium_error <= max(1e-5, abs(total_load) * 1e-6))
    if not equilibrium_ok:
        report_text += (
            "\n\nWARNING: Equilibrium check did not pass. "
            f"Difference = {equilibrium_error:.6g} kN. Treat this result as unverified."
        )

    return {
        "report": report_text,
        "equilibrium_ok": equilibrium_ok,
        "equilibrium_error_kN": equilibrium_error,
        "reactions": solution.reactions,
    }
