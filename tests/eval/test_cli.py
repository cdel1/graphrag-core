from typer.testing import CliRunner

from graphrag_core.eval.cli import app

runner = CliRunner()


def test_cli_help_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "eval" in result.stdout.lower()


def test_cli_run_subcommand_requires_pair_argument() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_cli_rebaseline_subcommand_exists() -> None:
    result = runner.invoke(app, ["rebaseline", "--help"])
    assert result.exit_code == 0


def test_cli_list_works_with_no_pairs() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
