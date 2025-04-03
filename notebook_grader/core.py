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

def create_grading_markdown_cell(problems: List[Dict[str, str]]) -> Dict[str, Any]:
    """Create a markdown cell containing the grading information."""
    severity_icons = {
        'low': '⚠️',
        'medium': '⛔',
        'high': '🚫',
        'critical': '❌'
    }

    markdown_content = [
        '<div style="background-color: #f8f8f8; border-left: 4px solid #ff6b6b; padding: 10px; margin: 10px 0;">',
        '<h4 style="color: #333; margin-top: 0;">📝 notebook-grader feedback for the above cell</h4>'
    ]

    for problem in problems:
        severity = problem['severity']
        icon = severity_icons.get(severity, '❗')
        markdown_content.extend([
            f'<div style="margin-left: 10px;">',
            f'{icon} <strong style="color: #e74c3c;">{severity.upper()}</strong>: {problem["type"]}',
            f'<p style="margin-left: 20px; color: #666;">{problem["description"]}</p>',
            '</div>'
        ])

    markdown_content.append('</div>')

    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': '\n'.join(markdown_content)
    }

def grade_notebook(*, notebook_path_or_url: str, model: str | None = None, vision_model: str | None=None, log_file: str | Path | None = None, auto: bool = False, output_notebook: Path | None = None) -> GradeNotebookResult:
    # Store all problems identified throughout the notebook
    all_problems: List[Dict[str, str]] = []
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

    # Initialize conversation with system prompt
    system_prompt = read_system_prompt()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

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
            user_message_content = create_user_message_content_for_cell(cells[cell_index])
            messages.append({"role": "user", "content": user_message_content})
            assistant_response, messages2, prompt_tokens, completion_tokens = run_completion_with_retries(
                messages=messages,
                model=model,
                num_retries=5
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            for a in user_message_content:
                if a['type'] == 'text':
                    print(a['text'])
                elif a['type'] == 'image_url':
                    num_bytes = len(a['image_url']['url'])
                    print(f"ATTACHED IMAGE {num_bytes / 1000} kilobytes")
            print(f'Prompt tokens: {total_prompt_tokens}, Completion tokens: {total_completion_tokens}')
            messages = messages2
            print('Assistant response:')
            problems = parse_assistant_response(assistant_response)

            if problems is None:
                print("No notebook_grader XML tags found in response for this cell.")
            elif not problems:
                print("No problems identified in this cell.")
            else:
                print('\n****************************************')
                print('****************************************')
                print("Identified problems in this cell:")
                for problem in problems:
                    print(f"\n[{problem['severity'].upper()}] {problem['type']}")
                    print(f"Description: {problem['description']}")
                    # Store problem with cell number for later display
                    problem_with_cell = problem.copy()
                    problem_with_cell['cell_number'] = str(cell_index + 1)
                    all_problems.append(problem_with_cell)

                # Add grading cell if problems were found
                if output_notebook_obj:
                    grading_cell = create_grading_markdown_cell(problems)
                    output_notebook_cells: List[Dict[str, Any]] = output_notebook_obj['cells']
                    output_notebook_cells.insert(output_cell_index + 1, grading_cell)
                    output_cell_index += 1

            # Write the output notebook if it was requested
            if output_notebook:
                with open(str(output_notebook), 'w') as f:
                    json.dump(output_notebook_obj, f, indent=2)

            # Print problem counts after cell evaluation
            severity_counts = {}
            for p in all_problems:
                severity = p['severity'].lower()
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            print(f'Total problems identified: {len(all_problems)}')
            for severity, count in severity_counts.items():
                print(f'{severity.upper()}: {count}')

            # press any key to continue
            if not auto:
                input("Press Enter to continue...")
            print('')

            output_cell_index += 1
    finally:
        # Restore original stdout and close log file
        if log_file_handle:
            sys.stdout = original_stdout
            log_file_handle.close()

    # Write the final output notebook if it was requested (really only necessary if there were no cells)
    if output_notebook and output_notebook_obj:
        with open(str(output_notebook), 'w') as f:
            json.dump(output_notebook_obj, f, indent=2)

    # Display final problem totals
    if all_problems:
        severity_counts = {}
        for p in all_problems:
            severity = p['severity'].lower()
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        print('\nFinal problem totals:')
        print(f'Total problems identified: {len(all_problems)}')
        for severity, count in severity_counts.items():
            print(f'{severity.upper()}: {count}')

        print('\nAll identified problems:')
        for problem in all_problems:
            print(f"\nCell {problem['cell_number']}:")
            print(f"[{problem['severity'].upper()}] {problem['type']}")
            print(f"Description: {problem['description']}")
    print('')  # Add blank line for readability

    return GradeNotebookResult(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_vision_prompt_tokens=total_vision_prompt_tokens,
        total_vision_completion_tokens=total_vision_completion_tokens,
        output_notebook_path=output_notebook if output_notebook else None
    )

def parse_assistant_response(response: str) -> List[Dict[str, str]] | None:
    """Parse the XML response from the assistant.

    Args:
        response: The raw response string from the assistant

    Returns:
        List of problems, each containing type, description, and severity,
        or None if no notebook_grader tags were found
    """
    # Find the content between notebook_grader tags
    start_tag = "<notebook_grader>"
    end_tag = "</notebook_grader>"

    start_idx = response.find(start_tag)
    if start_idx == -1:
        return None

    end_idx = response.find(end_tag, start_idx)
    if end_idx == -1:
        return None

    # Extract and clean the XML content
    xml_content = response[start_idx:end_idx + len(end_tag)]

    problems = []
    # Look for problem blocks
    while "<problem>" in xml_content:
        prob_start = xml_content.find("<problem>")
        prob_end = xml_content.find("</problem>", prob_start)
        if prob_end == -1:
            break

        problem_xml = xml_content[prob_start:prob_end + len("</problem>")]

        # Extract problem details
        def extract_field(field_name: str) -> str:
            start = problem_xml.find(f"<{field_name}>")
            if start == -1:
                return ""
            start += len(f"<{field_name}>")
            end = problem_xml.find(f"</{field_name}>", start)
            if end == -1:
                return ""
            return problem_xml[start:end].strip()

        problem = {
            "type": extract_field("type"),
            "description": extract_field("description"),
            "severity": extract_field("severity")
        }

        if all(problem.values()):  # Only add if all fields are present
            problems.append(problem)

        xml_content = xml_content[prob_end + len("</problem>"):]

    return problems

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
