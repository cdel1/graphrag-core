import os

import pytest
from typer.testing import CliRunner

from graphrag_core.eval.cli import app

runner = CliRunner()


def test_eval_list_includes_feverous() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "feverous" in result.stdout


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="FEVEROUS pair invokes the OpenAI LLM; set OPENAI_API_KEY to run.",
)
def test_eval_run_feverous_produces_a_run_report() -> None:
    result = runner.invoke(app, ["run", "feverous"])
    assert '"manifest_version"' in result.stdout
    assert "feverous@" in result.stdout
