from typing import Any
from system.api import generator_factory

MAX_NUM_REFLECTIONS = 3

EXECUTOR_PROMPT = """
You are a helpful assistant that executes the given request, and you'll be given:
1. The original request
2. The results and feedbacks of the answer from previous rounds, 
they will be wrapped in <previous_results_and_feedbacks> tags, and each set of result and feedback is wrapped in <result_and_feedback> tags.
If <previous_results_and_feedbacks> is empty, there is no previous results and feedbacks.

And please take the feedbacks into consideration to execute the request again, and put the results in <result> tags.
```
<original_request>
{original_request}
</original_request>

<previous_results_and_feedbacks>
{previous_results_and_feedbacks}
</previous_results_and_feedbacks>
```
"""

REFLECTOR_PROMPT = """
You are a helpful assistant that reviews the results and gives feedbacks, and you'll be given:
1. The original request
2. The results of the answer from another assistant

Please review the results and give feedbacks, and put the feedback in <feedback> tags.
Note that if the result is good enough, you can just return "good" as the feedback.

```
<original_request>
{original_request}
</original_request>

<result>
{result}
</result>
```
"""

def generate_executor_prompt(original_request: str, previous_results_and_feedbacks: list[tuple[str, str]]):
    return EXECUTOR_PROMPT.format(
        original_request=original_request,
        previous_results_and_feedbacks="\n".join(
            [
                f"<result_and_feedback>Results: {result}\nFeedback:     {feedback}\n</result_and_feedback>"
                for result, feedback in previous_results_and_feedbacks
            ]
        ),
    )

def generate_reflector_prompt(original_request: str, result: str):
    return REFLECTOR_PROMPT.format(
        original_request=original_request,
        result=result,
    )

def reflect(models: dict[str, Any], context: str, verbose: bool = False):
    """Reflect on the context and the results of the models.
    Args:
        models: Dictionary of model names to reflect on, and a final model to merge the results
        example: {"executor": "gpt-4o-mini", "reflector": "gpt-4o"}
        context: Input context for the pipeline
        verbose: Enable verbose output
    """
    assert "executor" in models
    assert "reflector" in models
    assert isinstance(models["executor"], str)
    assert isinstance(models["reflector"], str)

    executor = generator_factory(models["executor"], verbose)
    reflector = generator_factory(models["reflector"], verbose)

    not_good_enough = True
    num_rounds = 0
    previous_results_and_feedbacks = []
    status = ""
    while not_good_enough and num_rounds < MAX_NUM_REFLECTIONS:
        executor_prompt = generate_executor_prompt(context, previous_results_and_feedbacks)
        result, status = executor(executor_prompt)
        reflector_prompt = generate_reflector_prompt(context, result)
        feedback, _ = reflector(reflector_prompt)

        if feedback == "<feedback>good</feedback>":
            not_good_enough = False
        num_rounds += 1
        # update the previous results and feedbacks
        previous_results_and_feedbacks.append((result, feedback))

    return result, status
