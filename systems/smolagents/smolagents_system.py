# type: ignore
import os
import sys
import re
import fnmatch
from benchmark.benchmark_utils import print_error, print_warning

sys.path.append("./")

from benchmark.benchmark_api import System

class Smolagents(System):
    """
    A baseline system that uses a large language model (LLM) to process datasets and serve queries.
    """

    def __init__(self, model: str, name="baseline", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        print_warning("This system is a placeholder! Only use it with pre-computed cache results.")

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        print_warning("This system is a placeholder! Only use it with pre-computed cache results.")
        self.dataset_directory = dataset_directory

    def serve_query(self, query: str, query_id: str, subset_files: list|None) -> dict:
        print_warning("This system is a placeholder! Only use it with pre-computed cache results.")
        return { "answer": "This is a placeholder answer."}
