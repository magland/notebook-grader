import json
import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path
import requests
import copy
import os
from attr import dataclass

from .completion.run_completion import run_completion

def read_system_prompt() -> str:
    """Read and process the system prompt template."""
    template_path = Path(__file__).parent / "templates" / "system_prompt.txt"
    with open(template_path, "r") as f:
        content = f.read()
    return content

def read_user_message() -> str:
    """Read and process the user message template."""
    template_path = Path(__file__).parent / "templates" / "user_message.txt"
    with open(template_path, "r") as f:
        content = f.read()
    return content

class TeeOutput:
    """Class that duplicates output to both console and log file."""
    def __init__(self, log_file_handle):
        self.stdout = sys.stdout
        self.log_file = log_file_handle

    def write(self, text):
        self.stdout.write(text)
        if self.log_file:
            self.log_file.write(text)
            self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        if self.log_file:
            self.log_file.flush()

@dataclass
class GradeNotebookResult:
    total_prompt_tokens: int
    total_completion_tokens: int
    total_vision_prompt_tokens: int
    total_vision_completion_tokens: int
    output_notebook_path: Path | None = None
    output_json_path: Path | None = None

def parse_assistant_response(response: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Parse the assistant's response to extract rating and problems.

    Returns:
        Tuple containing:
        - Rating dict with 'quality' and 'rational' keys
        - List of problem dicts, each with 'type', 'description', and 'severity' keys
    """
    import re
    rating = {}
    problems = []

    # Extract rating block
    rating_match = re.search(r'<rating>\s*<quality>(.*?)</quality>\s*<rational>(.*?)</rational>\s*</rating>',
                           response, re.DOTALL)
    if rating_match:
        rating['quality'] = rating_match.group(1).strip()
        rating['rational'] = rating_match.group(2).strip()

    # Extract all problem blocks
    problem_matches = re.finditer(
        r'<problem>\s*<type>(.*?)</type>\s*<description>(.*?)</description>\s*<severity>(.*?)</severity>\s*</problem>',
        response, re.DOTALL
    )
    for match in problem_matches:
        problems.append({
            'type': match.group(1).strip(),
            'description': match.group(2).strip(),
            'severity': match.group(3).strip()
        })

    return rating, problems

def create_grading_markdown_cell(assistant_response: str) -> Dict[str, Any]:
    severity_icons = {
        'low': '⚠️',
        'medium': '⛔',
        'high': '🚫',
        'critical': '❌'
    }

    markdown_content = [
        '> ### 📝 notebook-grader feedback for above cell',
        '> ___'
    ]

    rating, problems = parse_assistant_response(assistant_response)

    if rating:
        markdown_content.extend([
            f'> **Cell Quality Rating**: ({rating["quality"]}/5)',
            '>',
            f'> **Rationale**: {rating["rational"]}'
        ])

    if problems:
        markdown_content.extend(['>', '> #### Identified Problems:'])
        for problem in problems:
            icon = severity_icons.get(problem['severity'].lower(), '❓')
            markdown_content.extend([
                '>',
                f'> **{icon} {problem["type"]}** (Severity: {problem["severity"]})',
                f'> {problem["description"]}'
            ])

    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': '\n'.join(markdown_content)
    }

def create_summary_markdown_cell(results: List[Dict[str, Any]], *, model: str, total_prompt_tokens: int, total_completion_tokens: int, notebook: Dict[str, Any]) -> Dict[str, Any]:
    """Create a markdown cell containing a summary of the grading results."""
    summary = calculate_grading_summary(results, notebook=notebook)

    severity_icons = {
        'low': '⚠️',
        'medium': '⛔',
        'high': '🚫',
        'critical': '❌'
    }

    from datetime import datetime

    markdown_content = [
        '> ### 📊 Notebook Grading Summary',
        '> ___',
        f'> **Total cells evaluated**: {len(results)}',
        '>',
        f'> **Total images in notebook**: {summary["total_images"]}',
        '>',
        f'> **Average cell rating**: {summary["average_rating"]}/5',
        '>',
        f'> **Total problems identified**: {summary["total_problems"]}',
        '>'
    ]

    for severity, count in summary["problems_by_severity"].items():
        if count > 0:
            icon = severity_icons.get(severity.lower(), '❓')
            markdown_content.append(f'>\n> {icon} **{severity.title()}**: {count}')
    markdown_content.append('>')
    markdown_content.append(f'> **Model used for grading**: {model}')
    markdown_content.append('>')
    markdown_content.append(f'> **Tokens used for grading**: {total_prompt_tokens} prompt + {total_completion_tokens} completion')
    markdown_content.append('>')
    markdown_content.append(f'> **Graded on**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    markdown_content.append('>')

    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': '\n'.join(markdown_content)
    }

def calculate_grading_summary(results: List[Dict[str, Any]], *, notebook: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics from grading results."""
    if not results:
        return {
            "average_rating": 0.0,
            "total_problems": 0,
            "problems_by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "total_images": 0
        }

    total_rating = 0
    total_problems = 0
    total_images = 0
    problems_by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}

    # Count total images in notebook
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for output in cell.get('outputs', []):
                if output['output_type'] in ['display_data', 'execute_result']:
                    if 'image/png' in output.get('data', {}):
                        total_images += 1

    for cell in results:
        # Sum ratings, converting string to float
        rating = cell["rating"].get("quality", "0")
        total_rating += float(rating)

        # Count problems by severity
        for problem in cell["problems"]:
            total_problems += 1
            severity = problem["severity"].lower()
            if severity in problems_by_severity:
                problems_by_severity[severity] += 1

    average_rating = total_rating / len(results)

    return {
        "average_rating": round(average_rating, 2),
        "total_problems": total_problems,
        "problems_by_severity": problems_by_severity,
        "total_images": total_images
    }

def grade_notebook(*, notebook_path_or_url: str, model: str | None = None, vision_model: str | None=None, log_file: str | Path | None = None, auto: bool = False, output_notebook: Path | None = None, output_json: Path) -> GradeNotebookResult:
    # Store all cell results identified throughout the notebook
    grading_results: List[Dict[str, Any]] = []
    """Perform a task based on the given instructions.

    Args:
        notebook_path_or_url: Path or URL to the notebook to be graded
        model: Optional model to use for completion
        vision_model: Optional model to use for vision tasks
        log_file: Optional file path to write verbose logs to
        auto: Whether to run in automatic mode where no user input is required
        output_notebook: Optional path to write the graded notebook to

    Returns:
        GradeNotebookResult containing token counts and output path
    """
    if not model:
        model = "google/gemini-2.0-flash-001"

    if not vision_model:
        vision_model = model

    # If it's a notebook in a GitHub repo then translate the notebook URL to raw URL
    if notebook_path_or_url.startswith('https://github.com/'):
        notebook_path_or_url = notebook_path_or_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

    # If notebook_path_or_url is a URL, download the notebook and read it as JSON
    # otherwise, read the file directly
    if notebook_path_or_url.startswith("http://") or notebook_path_or_url.startswith("https://"):
        response = requests.get(notebook_path_or_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download notebook from {notebook_path_or_url}")
        notebook_content = response.content.decode('utf-8')
        notebook = json.loads(notebook_content)
    else:
        with open(notebook_path_or_url, "r") as f:
            notebook_content = f.read()
        notebook = json.loads(notebook_content)

    if not 'cells' in notebook:
        raise Exception(f"Invalid notebook format. No cells found in the notebook.")

    # Create a copy of the notebook for output if requested
    output_notebook_obj = None
    output_cell_index = 0
    if output_notebook:
        # Delete output notebook if it exists
        if os.path.exists(output_notebook):
            os.remove(output_notebook)
        output_notebook_obj = copy.deepcopy(notebook)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_vision_prompt_tokens = 0
    total_vision_completion_tokens = 0
    cells = notebook['cells']

    # Open log file if specified and set up output redirection
    log_file_handle = open(log_file, 'w') if log_file else None
    original_stdout = sys.stdout

    try:
        if log_file_handle:
            sys.stdout = TeeOutput(log_file_handle)
        # Main conversation loop
        for cell_index in range(len(cells)):
            print(f'Grading cell {cell_index + 1} / {len(cells)}')

            # Initialize conversation with system prompt
            system_prompt = read_system_prompt()
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]

            previous_cells_message_content = [
                {'type': 'text', 'text': 'Here are the previous cells in the notebook for content'}
            ]

            for cell in cells[:cell_index]:
                previous_cells_message_content = previous_cells_message_content + create_user_message_content_for_cell(cell)

            messages.append({"role": "system", "content": previous_cells_message_content})

            current_cell_message_content: List[Dict[str, Any]] = [
                {'type': 'text', 'text': 'Here is the current cell in the notebook for content'}
            ]

            current_cell_message_content = current_cell_message_content + create_user_message_content_for_cell(cells[cell_index])

            messages.append({"role": "system", "content": current_cell_message_content})

            user_message = """
Please evaluate the current cell as you have been instructed.
"""

            messages.append({"role": "user", "content": user_message})

            assistant_response, _, prompt_tokens, completion_tokens = run_completion_with_retries(
                messages=messages,
                model=model,
                num_retries=5
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            # Parse and store results for this cell
            rating, problems = parse_assistant_response(assistant_response)
            cell_result = {
                "cell_index": cell_index,
                "rating": rating,
                "problems": problems
            }
            grading_results.append(cell_result)

            if output_notebook_obj:
                grading_cell = create_grading_markdown_cell(assistant_response)
                output_notebook_cells: List[Dict[str, Any]] = output_notebook_obj['cells']
                output_notebook_cells.insert(output_cell_index + 1, grading_cell)
                output_cell_index += 1

            for a in current_cell_message_content:
                if a['type'] == 'text':
                    print(a['text'])
                elif a['type'] == 'image_url':
                    num_bytes = len(a['image_url']['url'])
                    print(f"ATTACHED IMAGE {num_bytes / 1000} kilobytes")
            print('')
            print(f'Prompt tokens: {total_prompt_tokens}, Completion tokens: {total_completion_tokens}')
            print('Assistant response:')
            print(assistant_response)

            # Print grading summary after each cell
            summary = calculate_grading_summary(grading_results, notebook=notebook)
            print("\nGrading Summary:")
            print(f"Average Cell Rating: {summary['average_rating']:.2f}/5")
            print(f"Total Problems: {summary['total_problems']}")
            for severity, count in summary['problems_by_severity'].items():
                if count > 0:
                    print(f"- {severity.title()}: {count}")
            print()

            # Write the output notebook if it was requested
            if output_notebook:
                with open(str(output_notebook), 'w') as f:
                    json.dump(output_notebook_obj, f, indent=2)

            output_cell_index += 1

            if not auto:
                # wait for keyboard input
                input("Press Enter to continue...")
    finally:
        # Restore original stdout and close log file
        if log_file_handle:
            sys.stdout = original_stdout
            log_file_handle.close()

    # Write the final output notebook if it was requested (really only necessary if there were no cells)
    if output_notebook and output_notebook_obj and grading_results:
        summary_cell = create_summary_markdown_cell(grading_results, model=model, total_prompt_tokens=total_prompt_tokens, total_completion_tokens=total_completion_tokens, notebook=notebook)
        output_notebook_obj['cells'].insert(0, summary_cell)
        with open(str(output_notebook), 'w') as f:
            json.dump(output_notebook_obj, f, indent=2)

    # Get final summary including total image count for JSON output
    summary = calculate_grading_summary(grading_results, notebook=notebook)

    # Write aggregated results to JSON
    aggregate_results = {
        "notebook_path": notebook_path_or_url,
        "total_cells": len(cells),
        "total_images": summary["total_images"],
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_vision_prompt_tokens": total_vision_prompt_tokens,
        "total_vision_completion_tokens": total_vision_completion_tokens,
        "cells": grading_results
    }

    with open(output_json, "w") as f:
        json.dump(aggregate_results, f, indent=2)

    return GradeNotebookResult(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_vision_prompt_tokens=total_vision_prompt_tokens,
        total_vision_completion_tokens=total_vision_completion_tokens,
        output_notebook_path=output_notebook if output_notebook else None,
        output_json_path=output_json
    )

def create_user_message_content_for_cell(cell: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create user message content for a given cell."""
    content: List[Dict[str, Any]] = []
    if cell['cell_type'] == 'markdown':
        markdown_source = cell['source']
        content.append(
            {'type': 'text', 'text': 'INPUT-MARKDOWN: ' + ''.join(markdown_source)}
        )
    elif cell['cell_type'] == 'code':
        code_source = cell['source']
        content.append({'type': 'text', 'text': 'INPUT-CODE: ' + ''.join(code_source)})
        for x in cell['outputs']:
            output_type = x['output_type']
            if output_type == 'stream':
                content.append({'type': 'text', 'text': 'OUTPUT-TEXT: ' + '\n'.join(x['text'])})
            elif output_type == 'display_data' or output_type == 'execute_result':
                if 'image/png' in x['data']:
                    png_base64 = x['data']['image/png']
                    image_data_url = f"data:image/png;base64,{png_base64}"
                    content.append({'type': 'image_url', 'image_url': {'url': image_data_url}})
                else:
                    print(f'Warning: got output type {output_type} but no image/png data')
            else:
                print(f'Warning: unsupported output type {output_type}')
    else:
        print(f'Warning: unsupported cell type {cell["cell_type"]}')
        content.append(
            {'type': 'text', 'text': 'Unsupported cell type'}
        )
    return content

def run_completion_with_retries(
        messages: List[Dict[str, Any]], *,
        model: str,
        num_retries: int
    ) -> Tuple[str, List[Dict[str, Any]], int, int]:
    """Run completion with retries in case of failure."""
    import time
    retry_wait_time = 1
    for i in range(num_retries):
        try:
            return run_completion(messages, model=model)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error running completion: {e}")
            print(f"Retrying in {retry_wait_time} seconds...")
            time.sleep(retry_wait_time)
            retry_wait_time *= 2
    raise Exception(f"Failed to run completion after {num_retries} retries")
