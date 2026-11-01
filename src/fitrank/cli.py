"""The ``fitrank`` command: run, validate, list and predict from the terminal.

Exit codes: 0 success, 1 configuration error, 2 data error, 3 every candidate
failed. User errors are shown in red without a traceback (unless --verbose).
The leaderboard goes to stdout; the progress bar, warnings and errors go to
stderr.

"""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from sklearn.pipeline import Pipeline

from fitrank import __version__
from fitrank.benchmark import Benchmark
from fitrank.data import load_dataframe
from fitrank.evaluation import CandidateResult, display_name, higher_is_better
from fitrank.exceptions import DataError, FitRankError
from fitrank.reduction import list_reductions
from fitrank.registry import TASKS, list_models
from fitrank.results import Results, load_model

EXIT_CONFIG = 1
EXIT_DATA = 2
EXIT_ALL_FAILED = 3

app = typer.Typer(
    help="Train, compare and rank scikit-learn models from a YAML config.",
    no_args_is_help=True,
    add_completion=False,
)
list_app = typer.Typer(help="List the names you can use in a config.", no_args_is_help=True)
app.add_typer(list_app, name="list")

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)


@dataclass
class Options:
    verbose: bool = False
    quiet: bool = False


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"fitrank {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show each candidate, and tracebacks.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Only show the leaderboard.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Train, compare and rank scikit-learn models from a YAML config."""
    if verbose and quiet:
        raise typer.BadParameter("use --verbose or --quiet, not both")
    ctx.obj = Options(verbose=verbose, quiet=quiet)


@app.command()
def run(
    ctx: typer.Context,
    config: Annotated[Path, typer.Argument(help="The YAML config file.")],
) -> None:
    """Evaluate every candidate, show the leaderboard and save the run folder."""
    options: Options = ctx.obj
    with _user_errors(options):
        results = _run(Benchmark.from_yaml(config), options)

    if not console.is_terminal:  # a file or a pipe: no terminal width to fit in
        console.width = max(console.width, 160)
    console.print(_leaderboard_table(results))
    best = results.best
    if best is None:
        err_console.print(Text("Error: every candidate failed (see the error column)", "red"))
    elif not options.quiet:
        score = f"{display_name(results.metric)} = {best.cv_mean:.4f}"
        console.print(f"Best model: {best.model_name} + {best.reduction_name} ({score})")
    if not options.quiet and results.run_dir is not None:
        console.print(f"Results saved to {results.run_dir.as_posix()}/")
    if best is None:
        raise typer.Exit(EXIT_ALL_FAILED)


@app.command()
def validate(
    ctx: typer.Context,
    config: Annotated[Path, typer.Argument(help="The YAML config file.")],
) -> None:
    """Check the config and the data without training anything."""
    with _user_errors(ctx.obj):
        bench = Benchmark.from_yaml(config)
        plan = bench.validate()
    cfg = bench.config
    X = plan.X
    console.print(f"Config OK: {config.as_posix()}")
    console.print(f"Data OK: {len(X)} rows, {X.shape[1]} features, target {cfg.data.target!r}")
    console.print(
        f"{len(plan.candidates)} candidates: {len(cfg.reduction)} reduction(s)"
        f" x {len(cfg.models)} model(s), {cfg.evaluation.cv}-fold cross-validation"
    )


@list_app.command("models")
def list_models_command(
    task: Annotated[
        str | None, typer.Option(help="classification or regression (default: both).")
    ] = None,
) -> None:
    """List the models available for each task."""
    if task is not None and task not in TASKS:
        raise typer.BadParameter(f"must be one of: {', '.join(TASKS)}", param_hint="--task")
    for name in TASKS if task is None else (task,):
        models = list_models(name)
        typer.echo(f"{name} ({len(models)} models)")
        for model in models:
            typer.echo(f"  {model}")


@list_app.command("reductions")
def list_reductions_command() -> None:
    """List the dimensionality reductions."""
    for name in list_reductions():
        typer.echo(name)


@app.command()
def predict(
    ctx: typer.Context,
    model: Annotated[Path, typer.Argument(help="A best_model.joblib saved by fitrank run.")],
    data: Annotated[
        Path, typer.Argument(help="CSV, Excel or Parquet file with the same feature columns.")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="CSV file to write (default: print to stdout)."),
    ] = None,
) -> None:
    """Predict new data with a saved model; adds a 'prediction' column."""
    options: Options = ctx.obj
    with _user_errors(options):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = load_model(model)
        for warning in caught:
            err_console.print(Text(f"Warning: {warning.message}", "yellow"))
        frame = load_dataframe(data)
        predictions = _predict(pipeline, frame, data)

    table = frame.assign(prediction=predictions)
    if output is None:
        typer.echo(table.to_csv(index=False), nl=False)
        return
    table.to_csv(output, index=False)
    if not options.quiet:
        console.print(f"{len(table)} predictions written to {output.as_posix()}")


@contextmanager
def _user_errors(options: Options) -> Iterator[None]:
    """Show FitRank errors in red, without a traceback, and exit with their code."""
    try:
        yield
    except FitRankError as exc:
        if options.verbose:
            err_console.print_exception()
        err_console.print(Text(f"Error: {exc}", "red"))
        raise typer.Exit(EXIT_DATA if isinstance(exc, DataError) else EXIT_CONFIG) from None


def _run(bench: Benchmark, options: Options) -> Results:
    total = bench.n_candidates
    columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
    with Progress(
        *columns, TimeElapsedColumn(), console=err_console, transient=True, disable=options.quiet
    ) as progress:
        task = progress.add_task(f"Evaluating {total} candidates", total=total)

        def on_result(result: CandidateResult) -> None:
            progress.advance(task)
            if options.verbose:
                progress.console.print(Text(_describe(result)))

        return bench.run(on_result=on_result)


def _describe(result: CandidateResult) -> str:
    name = f"{result.reduction_name} + {result.model_name}"
    if result.status == "failed":
        return f"failed  {name}: {result.error}"
    score = f"cv {result.cv_mean:.4f} ± {result.cv_std:.4f}"
    return "; ".join([f"ok      {name}: {score} ({result.fit_seconds:.1f} s)", *result.notes])


def _leaderboard_table(results: Results) -> Table:
    name = display_name(results.metric)
    direction = "higher" if higher_is_better(results.metric) else "lower"
    table = Table(title=f"{results.config.project.name}: cv {name}, {direction} is better")
    for column in ("#", "model", "reduction", f"cv {name}", "cv std", f"test {name}", "time"):
        table.add_column(column, justify="left" if column in ("model", "reduction") else "right")
    table.add_column("best params / error")

    for row in results.leaderboard().to_dict("records"):
        if row["status"] == "ok":
            scores = [f"{row[column]:.4f}" for column in (f"cv_{name}", "cv_std", f"test_{name}")]
            detail: Text | str = _params(row["best_params"])
        else:
            scores, detail = ["failed", "", ""], Text(str(row["error"]), "red")
        rank, time = str(row["rank"]), f"{row['fit_s']:.1f} s"
        table.add_row(rank, row["model"], row["reduction"], *scores, time, detail)
    return table


def _params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _predict(pipeline: Pipeline, frame: pd.DataFrame, source: Path) -> Any:
    where = f"cannot predict on {source.as_posix()}"
    expected = [str(c) for c in getattr(pipeline, "feature_names_in_", [])]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise DataError(
            where, [f"missing column(s) the model was trained on: {', '.join(missing)}"]
        )
    try:
        # Extra columns, such as the target, are ignored.
        return pipeline.predict(frame[expected] if expected else frame)
    except ValueError as exc:
        raise DataError(where, [str(exc).strip().splitlines()[0]]) from None
