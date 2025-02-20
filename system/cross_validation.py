from typing import Any

from system.api import generator_factory


MERGER_PROMPT = """
You are a helpful assistant that merges the results of the models.

You will be given the original request, and the results of the models.

You need to:
1. Take the results of other models as the hints to the original request.
2. Merge the results into a single result that combines the best of all the results.
3. Return the merged result in the same format as the original request. The final results still need to be concise.
4. Please put the final results in <merged_result> tags.

Below is the original request and the results of the other models:

<original_request>
{original_request}
</original_request>

<results_of_other_models>
{results_of_other_models}
</results_of_other_models>
"""


def format_prompt_for_merger(original_request: str, results_of_other_models: list[str]) -> str:
    # Wrap each result in <result> tags
    formatted_results = [f"<result>{result}</result>" for result in results_of_other_models]
    return MERGER_PROMPT.format(
        original_request=original_request,
        results_of_other_models="\n".join(formatted_results)
    )

# TODO: We need to set higher temperature for the suggesters models,
# and lower temperature for the merger model.
def cross_validate(models: dict[str, Any], context: str, verbose: bool = False):
    """Cross validate the models by running them on the context and merging the results.
    Args:
        models: Dictionary of model names to cross validate, and a final model to merge the results
        example: {"suggesters": ["gpt-4o-mini", "gpt-4o-v", "mixral"], "merger": "gpt-4o-mini"}
        context: Input context for the pipeline
        verbose: Enable verbose output
    """
    # run the models
    # merge the results
    # return the results
    assert "suggesters" in models
    assert "merger" in models
    assert isinstance(models["suggesters"], list)
    assert isinstance(models["merger"], str)

    # run the models
    results = []
    for model in models["suggesters"]:
        generator = generator_factory(model, verbose)
        results.append(generator(context))

    merger_prompt = format_prompt_for_merger(context, results)
    merger = generator_factory(models["merger"], verbose)
    merged_result, status = merger(merger_prompt)
    return merged_result, status

