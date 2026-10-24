"""Reload the notebook's selected posterior after losing Python kernel state.

Only literal settings are read from the notebook; no notebook cell is executed.
This module cannot launch fitting, reset a run, or select another saved source.
"""
import ast
import json
from pathlib import Path

from .real_workflow import REPO


KEYS = ("SOURCE_ID", "RUN_NAME", "CANDIDATE_ID", "MCMC_INCLINATION_DEG",
        "MCMC_INCLINATION_SIGMA_DEG", "RESULTS", "VELOCITY_RADIUS_PC")
OPTIONAL_DEFAULTS = {"MCMC_INCLINATION_DEG": None,
                     "MCMC_INCLINATION_SIGMA_DEG": 10.,
                     "RESULTS": None, "VELOCITY_RADIUS_PC": None}


def _literal(node):
    # RESULTS may be a Path; constructing it does not read or execute a file.
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "Path" and len(node.args) == 1 and not node.keywords):
        return Path(ast.literal_eval(node.args[0]))
    return ast.literal_eval(node)


def saved_settings(notebook_path=None):
    path = (Path(notebook_path) if notebook_path is not None else
            REPO / "notebooks" / "legacy" / "rotation_only.ipynb")
    notebook = json.loads(path.read_text())
    # The minimal notebook only selects a candidate. Older notebooks may still
    # explicitly select a scan-prior run; honor those literals when present.
    values = dict(OPTIONAL_DEFAULTS)
    for cell in notebook["cells"]:
        if cell.get("id") not in ("three-file-settings", "candidate-selection"):
            continue
        source = "".join(cell["source"])
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in KEYS:
                    try:
                        values[target.id] = _literal(node.value)
                    except (ValueError, TypeError, SyntaxError) as exc:
                        raise ValueError(f"Run the settings and selection cells first: {target.id} "
                                         "is an expression rather than a saved literal value.") from exc
    missing = set(KEYS) - values.keys()
    if missing:
        raise ValueError(f"Run the settings and selection cells first; saved settings missing {sorted(missing)}.")
    return values


def load_for_review(namespace, *, notebook_path=None):
    """Return (saved result or None, radius_pc) without requiring prior cells.

    Existing kernel settings take precedence. Missing values come from the
    saved notebook's settings/selection cells, never a hardcoded source ID or
    the newest directory. Scientific-selection mismatches are not suppressed.
    """
    radius = namespace.get("VELOCITY_RADIUS_PC")
    try:
        loader = namespace.get("load_selected_posterior")
        if callable(loader):
            result = loader()
        else:
            settings = ({} if all(k in namespace for k in KEYS)
                        else saved_settings(notebook_path))
            settings.update({key: namespace[key] for key in KEYS if key in namespace})
            radius = settings["VELOCITY_RADIUS_PC"]
            from . import inclination_posterior, three_file
            options = dict(run_name=settings["RUN_NAME"], results=settings["RESULTS"])
            if settings["MCMC_INCLINATION_DEG"] is not None:
                result = inclination_posterior.load(settings["SOURCE_ID"], **options,
                    inclination_deg=settings["MCMC_INCLINATION_DEG"],
                    inclination_sigma_deg=settings["MCMC_INCLINATION_SIGMA_DEG"])
            elif settings["CANDIDATE_ID"] is not None:
                result = three_file.load(settings["SOURCE_ID"], **options,
                                         candidate_id=settings["CANDIDATE_ID"])
            else:
                result = None
    except FileNotFoundError:
        print("No saved posterior for this selection yet. Check the source/run selection; "
              "this display cell does not start a fit.")
        return None, radius
    if result is None:
        print("Choose a candidate in the MCMC cell to review its saved posterior.")
    return result, radius
