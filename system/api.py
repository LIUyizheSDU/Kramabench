from system.baseline import generator_factory
from system.cross_validation import cross_validate
from system.reflection import reflect
from typing import Union, Any
RUN_TYPE = ["baseline", "cross_validation", "reflection"]

def run_pipeline(models: Union[str, list[str], dict[str, Any]], context: str, run_type: str, verbose: bool = False):
    """Run the pipeline with different configurations based on run type.

    Args:
        models: Model configuration that varies by type:
            - baseline: single model name (str)
            - cross_validation: list of model names to cross validate, and a final model to merge the results
            - reflection: one model for reflection, and another model to critic and decide to accept or reject the reflection
        context: Input context for the pipeline
        run_type: Type of run from RUN_TYPE
        verbose: Enable verbose output
        **kwargs: Additional arguments specific to each run type
    
    Returns:
        tuple of (results, statistics)
    """    
    if run_type == "baseline":
        if not isinstance(models, str):
            raise ValueError("baseline requires a single model name as string")
        generator = generator_factory(models, verbose)
        # TODO GV: there is a mismatch between the inputs here and the expected inputs from the __call__ of the baseline generator
        results, stats = generator(context)
        return results, stats
    elif run_type == "cross_validation":
        return cross_validate(models, context, verbose)
    elif run_type == "reflection":
        return reflect(models, context, verbose)
    else:
        raise ValueError(f"Invalid run type: {run_type}")


