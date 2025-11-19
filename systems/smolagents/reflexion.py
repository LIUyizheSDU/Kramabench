# type: ignore
import os
import sys
import re
sys.path.append("./")
from benchmark.benchmark_utils import print_error, print_warning

import json
import pandas as pd
import subprocess
from typeguard import typechecked
from typing import Dict, List

from benchmark.benchmark_api import System
import argparse
import threading
import time

from dotenv import load_dotenv
from text_inspector_tool import TextInspectorTool
from tools import list_filepaths, list_input_filepaths, CRITIQUE_AGENT_PROMPT_TEMPLATE

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    AgentLogger
)
load_dotenv()

class SmolagentsReflexion(System):
    def __init__(self, model: str, name="example", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dataset_directory = None  # TODO(SUT engineer): Update me
        self.model = model # claude-3-7-sonnet-latest
        custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
        model_params = {
            "model_id": model, #"claude-3-7-sonnet-latest",
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8192,
        }
        self.llm_code = LiteLLMModel(**model_params)
        self.llm_reason = LiteLLMModel(**model_params) # Reasoning model can be set to a different one
        self.verbose = kwargs.get("verbose", False)
       
        self.debug = False
        self.output_dir = kwargs.get("output_dir", os.path.join(os.getcwd(), "testresults"))
        # Ablation studies: add additional parameters
        self.number_sampled_rows = kwargs.get("number_sampled_rows", 100)
        self.max_steps = kwargs.get("max_steps", 20)
        self.text_limit = kwargs.get("text_limit", 100000)
        self.verbosity_level = kwargs.get("verbosity_level", 2)
        self.planning_interval = kwargs.get("planning_interval", 4)

        self.question_output_dir = None  # to be set in run()
        self.question_intermediate_dir = None  # to be set in run()
    
    def _init_output_dir(self, query_id: str) -> None:
        """
        Initialize the output directory for the question.
        :param question: Question object
        """
        question_output_dir = os.path.join(self.output_dir, query_id)
        if not os.path.exists(question_output_dir):
            os.makedirs(question_output_dir)
        self.question_output_dir = question_output_dir
        self.question_intermediate_dir = os.path.join(
            self.question_output_dir, "_intermediate"
        )
        if not os.path.exists(self.question_intermediate_dir):
            os.makedirs(self.question_intermediate_dir)
    
    def _parse_output(self, output: str, query_id: str) -> (str, str):
        """
        Parse the output from the agent to extract the answer and pipeline code.
        Write intermediate logs to file.
        :param output: str
        :param query_id: str
        :return: (answer, pipeline_code)
        """
        # Parse the output to extract the final answer and pipeline code
        try:
            # Case 1: agent already returned a dict-like object
            if isinstance(output, dict):
                output_dict = output
            else:
                # Coerce to string in case it's something like a custom object
                output_str = str(output)

                # Try to find the first {...} block (non-greedy) in the text
                match = re.search(r"\{.*?\}", output_str, re.DOTALL)
                if not match:
                    raise ValueError("No JSON-like object found in agent output.")

                obj_str = match.group(0)

                # First try strict JSON
                try:
                    output_dict = json.loads(obj_str)
                except json.JSONDecodeError:
                    # Fallback: many LLMs emit Python dict reprs with single quotes
                    try:
                        output_dict = ast.literal_eval(obj_str)
                        if not isinstance(output_dict, dict):
                            raise ValueError("Parsed object is not a dict.")
                    except Exception as inner_e:
                        raise ValueError(
                            f"Failed to parse as JSON or Python dict: {inner_e}"
                        ) from inner_e

            # Extract fields with sensible fallbacks / alternative keys
            answer = (
                output_dict.get("explanation")
                or output_dict.get("answer")
                or "No explanation found."
            )
            pipeline_code = (
                output_dict.get("pipeline_code")
                or output_dict.get("code")
                or "No pipeline code found."
            )

        except Exception as e:
            print_error(f"Failed to parse output JSON: {e}\nRaw output:\n{output}")
            answer = "Failed to parse output."
            pipeline_code = "N/A"
        
        # Write results to file
        with open(os.path.join(self.question_intermediate_dir, f"answer.txt"), "w") as f:
            f.write(answer)
        with open(os.path.join(self.question_intermediate_dir, f"pipeline_code.py"), "w") as f:
            f.write(pipeline_code)

        return answer, pipeline_code

    def create_agent(self, task_id:str) -> CodeAgent:

        logger_path = os.path.join(self.question_intermediate_dir, f"{task_id}.txt")
        logger = AgentLogger(level=self.verbosity_level, log_file=logger_path)
        #############################################
        
        critique_agent = ToolCallingAgent(
            model=self.llm_reason,
            tools=[TextInspectorTool(self.llm_reason, self.text_limit), AnswerInspectorTool(self.llm_reason), list_filepaths],
            max_steps=1,
            verbosity_level=self.verbosity_level,
            logger=logger,
            planning_interval=self.planning_interval,
            name="critique_agent",
            description="""
                An independent agent that acts as a critical reviewer for each step in a multi-step agent workflow.

                The critique_agent evaluates the most recent plan and corresponding tool output (observation) 
                produced by a primary agent, such as a CodeAgent. It provides constructive feedback, flags 
                issues, and suggests improvements or next actions without directly modifying the environment.

                This agent is intended to serve as a lightweight oversight mechanism in agentic systems, helping 
                to improve reliability, reduce errors, and promote clearer reasoning in autonomous workflows.

                Invoke this agent after each step of the primary agent to ensure that the actions taken are appropriate and well-reasoned.
            """,
            provide_run_summary=True,
        )
        critique_agent.prompt_templates["managed_agent"]["task"] += CRITIQUE_AGENT_PROMPT_TEMPLATE

        manager_agent = CodeAgent(
            model=self.llm_code,
            planner_model=self.llm_reason,
            tools=[TextInspectorTool(self.llm_reason, self.text_limit), list_input_filepaths], #process_beach_data
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            additional_authorized_imports=["*"],
            planning_interval=self.planning_interval,
            logger=logger,
            managed_agents=[critique_agent],
        )

        return manager_agent

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        # read files based on the dataset_directory, could be pdf, csv, etc.
        # store the data in a dictionary
        # store the dictionary in self.dataset_directory
        self.dataset_directory = dataset_directory
        self.dataset = {}
        # read the files
        for file in os.listdir(dataset_directory):
            if file.endswith(".csv"):
                self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file))

    @typechecked
    def serve_query(self, query: str, query_id: str = "default_name-0", subset_files:List[str]=[]) -> Dict:
        """
        Serve a query using the LLM.
        The query should be in natural language, and the response can be in either natural language or JSON format.
        :param query: str
        :param query_id: str
        :param subset_files: list of strings with filenames to include in the experiments
        :return: output_dict {"explanation": answer, "pipeline_code": pipeline_code}
        """
        # TODO: 
        self._init_output_dir(query_id)
        dataset_name = query_id.split("-")[0]
        task = f"""
            Workload name: {dataset_name}
            dataset_directory = {os.path.join(os.getcwd(), 'data', dataset_name, 'input')}
            Answer the question: {query}; Use the {dataset_name} dataset. 
            
            If you see unusual or unexpected intermediate results, use the critique agent to evaluate the step and provide feedback.
            DO NOT assume the correctness of the intermediate results and final results. Be skeptical and ensure they are correct by cross-checking with the dataset.
            Especially, if you see missing or incomplete intermediate results (e.g., nan or 0.0), question yourself if it's a coding issue, a logic design issue, or an actual data issue.
            After every step, ask yourself if every step is necessary and if the final answer is correct.
            **Report the final answer (just the answer, no explanation needed) AND the complete code pipeline (as a string) used to get there in your final response, in JSON format with keys "explanation" and "pipeline_code".**
            Example final response:
            {{"explanation": "42", "pipeline_code": "import pandas as pd\\n..."}}
        """
        
        start_time = time.time()
        agent = self.create_agent(task_id=query_id)
        output = agent.run(task)
        runtime = time.time() - start_time

        answer, pipeline_code = self._parse_output(output, query_id)
        results = {
            "id": query_id,
            "runtime": runtime,
            "explannation": answer,
            "pipeline_code": pipeline_code,
        }
        print(results)
        return results


def main():
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    questions = [
        {
            "id": "environment-easy-1",
            "query": "In 2018, how many bacterial exceedances were observed in freshwater beaches?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-medium-1",
            "query": "For the freshwater beaches, what is the difference between the percentage exceedance rate in 2023 and the historic average from 2002 to 2022?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-hard-1",
            "query": "For the marine beaches, what is the Pearson product-moment correlation from 2002 to 2023 between the rainfall amount in inches during the months June, July, August, and September and the percentage exceedance rate?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
    ]

    # Process each question
    out_dir = os.path.join(current_dir, "../../testresults/reflexion")
    smolagents_dr = SmolagentsReflexion(
        model="claude-3-7-sonnet-latest", output_dir=out_dir, verbose=True
    )
    for question in questions:
        print(f"Processing question: {question['id']}")
        # For debugging purposes, also input question.id
        smolagents_dr.serve_query(question["query"], question["id"])
        break

if __name__ == "__main__":
    main()