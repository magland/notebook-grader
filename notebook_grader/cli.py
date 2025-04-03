import click
from pathlib import Path
from .core import grade_notebook

@click.group()
def cli():
    """Command-line interface for executing tasks using large language models"""
    pass

@cli.command("grade-notebook")
@click.argument('notebook_path_or_url', required=True)
@click.option('--model', '-m', help='Model to use for completion (default: google/gemini-2.0-flash-001)')
@click.option('--vision-model', default=None, help='Model to use for vision tasks (default: same as --model)')
@click.option('--log-file', '-l', type=click.Path(dir_okay=False, path_type=Path), help='File to write verbose logs to')
@click.option('--auto', is_flag=True, help='Run in automatic mode where no user input is required')
@click.option('--output-notebook', type=click.Path(dir_okay=False, path_type=Path), help='Path to write the graded notebook to')
def grade_notebook_cmd(notebook_path_or_url: str, model: str | None, vision_model: str | None, log_file: Path | None, auto: bool, output_notebook: Path | None):
    """Grade a Python Jupyter notebook by identifying problems.

    Recommended OpenRouter Models:
        - google/gemini-2.0-flash-001
        - anthropic/claude-3.5-sonnet
        - anthropic/claude-3.7-sonnet
    """

    grade_notebook(
        notebook_path_or_url=notebook_path_or_url,
        model=model,
        vision_model=vision_model,
        log_file=log_file,
        auto=auto,
        output_notebook=output_notebook
    )

if __name__ == "__main__":
    cli()
