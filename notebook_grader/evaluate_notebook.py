import json
import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path
import requests
import copy
import os
from attr import dataclass

from notebook_grader.grade_notebook import create_user_message_content_for_cell

from .completion.run_completion import run_completion

def read_evaluate_system_prompt() -> str:
    """Read and process the system prompt template."""
    template_path = Path(__file__).parent / "templates" / "evaluate_system_prompt.txt"
    with open(template_path, "r") as f:
        content = f.read()
    return content

@dataclass
class EvaluateNotebookResult:
    total_prompt_tokens: int
    total_completion_tokens: int
    output_json_path: Path | None = None

def evaluate_notebook(*, notebook_path_or_url: str, model: str | None = None, auto: bool = False, output_json: Path) -> EvaluateNotebookResult:
    if not model:
        model = "google/gemini-2.0-flash-001"

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

    total_prompt_tokens = 0
    total_completion_tokens = 0
    cells = notebook['cells']

    system_prompt = read_evaluate_system_prompt()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]
    for cell in cells:
        content = create_user_message_content_for_cell(cell)
        messages.append({"role": "system", "content": content})

    user_message = """
Please provide a thorough evaluation of this notebook.
"""
    messages.append({"role": "user", "content": user_message})
    assistant_response, _, prompt_tokens, completion_tokens = run_completion(
        messages=messages,
        model=model
    )
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    print(assistant_response)
    print(f'Prompt tokens: {total_prompt_tokens}, Completion tokens: {total_completion_tokens}')
    print(f'Length of response: {len(assistant_response)}')

    results = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "assistant_response": assistant_response,
        'length_of_response': len(assistant_response)
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to {output_json}")
    return EvaluateNotebookResult(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        output_json_path=output_json
    )

