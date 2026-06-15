import os

import pytest
from typer.testing import CliRunner

from graphrag_core.eval.cli import app

runner = CliRunner()


def test_eval_list_includes_docred() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "docred" in result.stdout


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="DocRED pair invokes the OpenAI LLM; set OPENAI_API_KEY to run.",
)
def test_eval_run_docred_produces_a_run_report() -> None:
    result = runner.invoke(app, ["run", "docred"])
    # Tier-1 fails because Document nodes are emitted without Claims/SOURCED_FROM,
    # so the runner exits non-zero. The point of the smoke test is that the
    # harness loop runs end-to-end and emits a structured RunReport.
    assert '"manifest_version"' in result.stdout
    assert "docred@" in result.stdout
