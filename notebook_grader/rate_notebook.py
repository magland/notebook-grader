import json
import os
from typing import List, Dict, Any
from pathlib import Path
import requests
from attr import dataclass
import yaml

from notebook_grader.grade_notebook import create_user_message_content_for_cell

from .completion.run_completion import run_completion

def read_rate_system_prompt() -> str:
    """Read and process the system prompt template."""
    template_path = Path(__file__).parent / "templates" / "rate_system_prompt.txt"
    with open(template_path, "r") as f:
        content = f.read()
    return content

@dataclass
class RateNotebookResult:
    total_prompt_tokens: int
    total_completion_tokens: int
    output_json_path: str

def rate_notebook(*, notebook_path_or_url: str, model: str | None = None, auto: bool = False, questions_yaml: str, output_json: str) -> RateNotebookResult:
    num_repeats = 3

    # load questions
    with open(questions_yaml, "r") as f:
        questions = yaml.safe_load(f)

    assert 'questions' in questions, "questions.yaml must contain a 'questions' key"
    for question in questions['questions']:
        assert 'name' in question, "Each question must have a 'name' key"
        assert 'version' in question, "Each question must have a 'version' key"
        assert 'question' in question, "Each question must have a 'question' key"
        assert 'rubric' in question, "Each question must have a 'rubric' key"
        for rub in question['rubric']:
            assert 'score' in rub, "Each rubric must have a 'score' key"
            assert 'description' in rub, "Each rubric must have a 'description' key"

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

    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            existing_results = json.load(f)
            assert 'scores' in existing_results, "Existing results JSON must contain a 'scores' key"
            for score in existing_results['scores']:
                assert 'name' in score, "Each score must have a 'name' key"
                assert 'version' in score, "Each score must have a 'version' key"
                assert 'score' in score, "Each score must have a 'score' key"
                assert 'reps' in score, "Each score must have a 'reps' key"
                for rep in score['reps']:
                    assert 'score' in rep, "Each repetition must have a 'score' key"
                    assert 'thinking' in rep, "Each repetition must have a 'thinking' key"
                    assert 'repnum' in rep, "Each repetition must have a 'repnum' key"
    else:
        existing_results = {
            'scores': []
        }


    total_prompt_tokens = 0
    total_completion_tokens = 0
    cells = notebook['cells']

    new_results = {
        'scores': []
    }

    for question in questions['questions']:
        existing_score = None
        for existing_score0 in existing_results['scores']:
            if existing_score0['name'] == question['name'] and existing_score0['version'] == question['version']:
                if len(existing_score0['reps']) == num_repeats:
                    existing_score = existing_score0
                    print(f"Found existing score for question {question['name']} version {question['version']}: {existing_score0['score']}")
                    break
                else:
                    print(f"Found existing score for question {question['name']} version {question['version']}, but it has {len(existing_score0['reps'])} repetitions. Repeating the question.")
        if existing_score:
            new_results['scores'].append(existing_score)
            print(f"Skipping question {question['name']} version {question['version']} as it already exists in the results.")
            continue

        reps = []
        for repnum in range(num_repeats):
            system_prompt = read_rate_system_prompt()
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]
            for cell in cells:
                content = create_user_message_content_for_cell(cell)
                messages.append({"role": "system", "content": content})

            user_message = f"Please rate the notebook based on the following question: {question['question']}\n\n"
            user_message += f"Rubric:\n"
            for rub in question['rubric']:
                user_message += f"- {rub['score']}: {rub['description']}\n"
            user_message += """
    Remember that your output should be in the following format:

    <notebook_rater>
        <thinking>Your reasoning for the score</thinking>
        <score>numeric_score</score>
    </notebook_rater>
    """
            messages.append({"role": "user", "content": user_message})
            print(f"Rating question {question['name']} version {question['version']} Repetition {repnum + 1}/{num_repeats}")
            print(question['question'])
            assistant_response, _, prompt_tokens, completion_tokens = run_completion(
                messages=messages,
                model=model
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            print(assistant_response)
            print(f'Prompt tokens: {total_prompt_tokens}, Completion tokens: {total_completion_tokens}')

            a = parse_assistant_response(assistant_response)
            reps.append({
                'score': a['score'],
                'thinking': a['thinking'],
                'repnum': repnum
            })
        average_score = sum([rep['score'] for rep in reps]) / len(reps)
        print(f"Score: {average_score} : {[rep['score'] for rep in reps]}")
        new_results['scores'].append({
            'name': question['name'],
            'version': question['version'],
            'score': average_score,
            'reps': reps
        })

    with open(output_json, "w") as f:
        json.dump(new_results, f, indent=4)
    print(f"Results written to {output_json}")

    print('')
    # Print a summary of all the scores
    for question in new_results['scores']:
        print(question['name'])
        print(f"{question['score']:.2f} {[rep['score'] for rep in question['reps']]}")
        print('')

    # Report number of tokens used
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")

    return RateNotebookResult(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        output_json_path=output_json
    )

def parse_assistant_response(assistant_response: str) -> Dict[str, Any]:
    ind1 = assistant_response.find("<notebook_rater>")
    ind2 = assistant_response.find("</notebook_rater>")
    if ind1 == -1 or ind2 == -1:
        raise ValueError("Invalid assistant response format")
    ind1 += len("<notebook_rater>")
    content = assistant_response[ind1:ind2]
    thinking_ind1 = content.find("<thinking>")
    thinking_ind2 = content.find("</thinking>")
    if thinking_ind1 == -1 or thinking_ind2 == -1:
        raise ValueError("Invalid assistant response format")
    thinking_ind1 += len("<thinking>")
    thinking = content[thinking_ind1:thinking_ind2].strip()
    score_ind1 = content.find("<score>")
    score_ind2 = content.find("</score>")
    if score_ind1 == -1 or score_ind2 == -1:
        raise ValueError("Invalid assistant response format")
    score_ind1 += len("<score>")
    score = content[score_ind1:score_ind2].strip()
    try:
        score = float(score)
    except ValueError:
        raise ValueError("Invalid score format")
    return {
        "thinking": thinking,
        "score": score
    }

