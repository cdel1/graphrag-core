import os

import pytest
from typer.testing import CliRunner

from graphrag_core.eval.cli import app

runner = CliRunner()


def test_eval_list_includes_feverous() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "feverous" in result.stdout


def test_eval_list_includes_feverous_anthropic() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "feverous_anthropic" in result.stdout


# Both e2e tests below bill a real provider. `integration` is what keeps them out
# of a plain `pytest tests/`: the key-presence skipif says *which* credential is
# missing, but a key that happens to be exported is not a decision to spend. The
# marker is (tessera#586's lesson applied to money rather than to data).
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="FEVEROUS pair invokes the OpenAI LLM; set OPENAI_API_KEY to run.",
)
def test_eval_run_feverous_produces_a_run_report() -> None:
    result = runner.invoke(app, ["run", "feverous"])
    assert '"manifest_version"' in result.stdout
    assert "feverous@" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="feverous_anthropic pair invokes the Anthropic LLM; set ANTHROPIC_API_KEY to run.",
)
def test_eval_run_feverous_anthropic_produces_a_run_report() -> None:
    """The Anthropic pair had no test at all — the same loop, a second provider.

    `feverous_anthropic` has been registered since the pair landed, pinned to
    claude-sonnet-4-6, but nothing exercised it: both e2e tests gated on
    `OPENAI_API_KEY`, so the provider swap the harness exists to support was
    never proven end to end.

    The assertions are the same as the OpenAI pair's because the emitted report
    genuinely cannot be told apart from it: both load the same fixture, so both
    report `manifest_version: feverous@…`, and `RunReport` carries no
    `model_pin` — that field exists only on `BaselineFile`. Asserting a provider
    here would need a schema change, tracked separately.
    """
    result = runner.invoke(app, ["run", "feverous_anthropic"])
    assert '"manifest_version"' in result.stdout
    assert "feverous@" in result.stdout
    assert '"passed": true' in result.stdout
