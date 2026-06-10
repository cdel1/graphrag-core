"""`graphrag-core` Typer app for the eval harness.

Subcommands:
  run        — run the eval harness for a registered pair
  rebaseline — write current slice scores as the new baseline
  list       — list registered manifest/scorer pairs

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §4.1.
"""

from __future__ import annotations

import asyncio

import typer

app = typer.Typer(help="graphrag-core evaluation harness CLI")


@app.command()
def run(pair: str = typer.Argument(..., help="Registered pair name (e.g. 'docred')")) -> None:
    """Run the eval harness for PAIR; fail loud on slice regression."""
    from graphrag_core.eval.registry import build_default_components, get_pair

    pair_def = get_pair(pair)
    harness, _ = build_default_components(pair_def)
    report = asyncio.run(harness.run())
    typer.echo(report.model_dump_json(indent=2))
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def rebaseline(
    pair: str = typer.Argument(..., help="Registered pair name"),
) -> None:
    """Run the eval, persist its slice scores as the new baseline for PAIR."""
    from graphrag_core.eval.registry import (
        build_default_components,
        get_pair,
        persist_as_baseline,
    )

    pair_def = get_pair(pair)
    harness, baseline_store = build_default_components(pair_def)
    report = asyncio.run(harness.run())
    path = persist_as_baseline(report, pair_def, baseline_store)
    typer.echo(f"Wrote baseline to {path}")


@app.command(name="list")
def list_pairs() -> None:
    """List registered manifest/scorer pairs."""
    from graphrag_core.eval.registry import list_registered

    for name in list_registered():
        typer.echo(name)
