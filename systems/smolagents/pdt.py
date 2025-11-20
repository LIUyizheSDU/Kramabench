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
import ast
import time

from dotenv import load_dotenv
from .text_inspector_tool import TextInspectorTool
from .tools import list_input_filepaths, write_file, get_csv_metadata, summarize_dataframe
from .smolagents_utils import parse_token_counts

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
    AgentLogger
)
load_dotenv()

class SmolagentsPDT(System):
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
        if "claude" in model:
            reason_model_params = {
                "model_id": "o3", #"claude-3-7-sonnet-latest",
                "custom_role_conversions": custom_role_conversions,
                "max_completion_tokens": 8192,
            }
            self.llm_reason = LiteLLMModel(**reason_model_params) # Reasoning model can be set to a different one
        else: self.llm_reason = LiteLLMModel(**model_params) # same as code model
        self.verbose = kwargs.get("verbose", False)
       
        self.debug = False
        self.output_dir = kwargs.get("output_dir", os.path.join(os.getcwd(), "testresults"))
        # Ablation studies: add additional parameters
        self.number_sampled_rows = kwargs.get("number_sampled_rows", 100)
        self.max_steps = kwargs.get("max_steps", 10)
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
    
    def _get_output(self, answer_path: str, pipeline_code_path: str) -> (str, str):
        """
        Get the output from the answer and pipeline code files.
        :param answer_path: str
        :param pipeline_code_path: str
        :return: (answer, pipeline_code)
        """
        try:
            with open(answer_path, "r") as f:
                answer = f.read().strip()
        except Exception as e:
            print_error(f"Failed to read answer file: {e}")
            answer = "Failed to read answer."

        try:
            with open(pipeline_code_path, "r") as f:
                pipeline_code = f.read().strip()
        except Exception as e:
            print_error(f"Failed to read pipeline code file: {e}")
            pipeline_code = "Failed to read pipeline code."

        return answer, pipeline_code
    
    def _get_token_counts(self, query_id: str) -> Dict[str, int]:
        """
        Get the token counts from the log file.
        :param query_id: str
        :return: Dict[str, int]
        """
        logger_path = os.path.join(self.question_intermediate_dir, f"{query_id}.txt")
        with open(logger_path, "r") as f:
            trace = f.read()
        inp, out = parse_token_counts(trace)
        return {"input_tokens": inp, "output_tokens": out}
    
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

    def create_pdt_agents(self, task_id: str):

        logger_path = os.path.join(self.question_intermediate_dir, f"{task_id}.txt")
        logger = AgentLogger(level=self.verbosity_level, log_file=logger_path)

        # === Planner Agent ===
        planner_agent = ToolCallingAgent(
            model=self.llm_reason,
            tools=[list_input_filepaths],
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            logger=logger,
            planning_interval=self.planning_interval,
            name="planner_agent",
            description=(
                "A high-level planning agent that reads the user task and available data, "
                "then proposes a step-by-step PLAN (3–5 steps) for how to solve the task. "
                "The plan should be model-agnostic and expressed in natural language. "
                "Do NOT execute code or tools directly; just plan. "
                "Do NOT give any answer yet; just provide the PLAN."
            ),
            provide_run_summary=True,
        )

        # === Subtask Decomposer Agent ===
        decomposer_agent = ToolCallingAgent(
            model=self.llm_reason,
            tools=[list_input_filepaths, summarize_dataframe],
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            logger=logger,
            planning_interval=self.planning_interval,
            name="decomposer_agent",
            description=(
                "Given a user task and a high-level PLAN from the planner_agent, "
                "this agent decomposes the plan into a small sequence of atomic subtasks "
                "that a code executor can implement. It MUST output a JSON list of subtasks "
                "with the schema:\n\n"
                "[\n"
                "  {\"id\": 1, \"description\": \"...\", \"depends_on\": []},\n"
                "  {\"id\": 2, \"description\": \"...\", \"depends_on\": [1]},\n"
                "  ...\n"
                "]\n\n"
                "Each description should be concrete enough to be implemented in Python "
                "with access to the workload dataset.\n\n"
                "The list MUST contain between 1 and 5 subtasks (inclusive). "
                "If more than 5 steps seem necessary, merge related steps."
            ),
            provide_run_summary=True,
        )

        # === Tool/Code Executor Agent ===
        executor_agent = CodeAgent(
            model=self.llm_code,
            planner_model=self.llm_reason,
            tools=[
                list_input_filepaths,
                TextInspectorTool(self.llm_reason, self.text_limit),
                write_file,
            ],
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            additional_authorized_imports=["*"],
            planning_interval=self.planning_interval,
            logger=logger,
            name="executor_agent",
            description=(
                "A Python-executing agent that solves a SINGLE subtask at a time. "
                "Given the global task, the high-level plan, and the current subtask "
                "description, it should write and run Python code over the workload "
                "dataset to complete that subtask. It should print intermediate results, "
                "and return a concise textual summary of what was done and the key outputs. "
                "Use write_file tool to record the end-to-end code pipeline and final answer."
            ),
        )

        return planner_agent, decomposer_agent, executor_agent

    def run_pdt_pipeline(self,planner_agent, decomposer_agent, executor_agent, task_prompt, workload_name, task_id: str) -> Dict:
        """
        Orchestrates the Planner -> Decomposer -> Executor flow for a single question.
        Returns a dict with the plan, subtasks, subtask_outputs, and final_answer.
        """

        # ---- 1) Planner: high-level plan ----
        planner_prompt = f"""
            You are the planner_agent in a multi-agent PDT architecture.

            Workload name: {workload_name}
            User task:
            {task_prompt}

            Your job:
            1. Understand the task and the likely structure of the dataset(s).
            2. Propose a concrete, numbered PLAN with 3-5 steps (5 is the MAX).
            3. Focus on data discovery, cleaning, joining, aggregation, and analysis.
            4. Do NOT write code. Do NOT execute tools. Only output a plan in natural language.

            Output format:
            - Start with a short 1–2 sentence summary of the task.
            - Then list your steps as a numbered list, e.g.
            1) ...
            2) ...
        """
        #high_level_plan = planner_agent.run(planner_prompt)

        # ---- 2) Decomposer: JSON subtasks ----
        decomposer_prompt = f"""
            You are the decomposer_agent in a Planner → Subtask Decomposer → Tool/Code Executor architecture.

            Workload name: {workload_name}

            User task:
            {task_prompt}

            Decompose this into a sequence of atomic subtasks that a code agent can execute.
            Each subtask should:
            - be as independent and concrete as possible,
            - be implementable by executing Python over the workload data,
            - have explicit dependencies.
            - 5 is the MAX number of subtasks you can create. 

            STRICTLY output valid JSON with this schema and nothing else:

            [
            {{"id": 1, "description": "...", "depends_on": []}},
            {{"id": 2, "description": "...", "depends_on": [1]}},
            ...
            ]
        """
        raw_subtasks = decomposer_agent.run(decomposer_prompt)
        token_counts = self._get_token_counts(task_id)
        # Try to parse JSON (you can add more robust cleaning if needed)
        try:
            subtasks = json.loads(str(raw_subtasks))
        except json.JSONDecodeError:
            # Fallback: wrap in list if model forgot brackets etc.
            try:
                cleaned = raw_subtasks.strip()
                if not cleaned.startswith("["):
                    cleaned = "[" + cleaned
                if not cleaned.endswith("]"):
                    cleaned = cleaned + "]"
                subtasks = json.loads(cleaned)
            except Exception:
                # If this still fails, just fall back to a single-subtask wrapper
                subtasks = [
                    {
                        "id": 1,
                        "description": f"Execute the entire task directly: {task_prompt}",
                        "depends_on": [],
                    }
                ]

        # ---- 3) Executor: run each subtask sequentially ----
        answer_path = os.path.join(self.question_intermediate_dir, f"answer.txt")
        pipeline_code_path = os.path.join(self.question_intermediate_dir, f"pipeline_code.py")
        subtask_outputs = {}
        last_id = sorted(subtask_outputs.keys())[-1] if subtask_outputs else 1
        for subtask in subtasks:
            sid = subtask.get("id")
            desc = subtask.get("description", "")
            deps = subtask.get("depends_on", [])

            deps_summaries = [
                f"(Subtask {d}): {subtask_outputs.get(d, '')}" for d in deps if d in subtask_outputs
            ]
            deps_text = "\n".join(deps_summaries) if deps_summaries else "None yet."

            executor_prompt = f"""
            You are the executor_agent in a Planner → Decomposer → Executor (PDT) architecture.

            Global workload name: {workload_name}

            Global user task:
            {task_prompt}

            You are now executing Subtask {sid}.

            Subtask {sid} description:
            \"\"\"{desc}\"\"\"

            Summaries of completed dependency subtasks:
            {deps_text}

            Your job:
            - Focus ONLY on Subtask {sid}.
            - Use Python code (and the available tools) to complete this subtask over the workload dataset.
            - Print intermediate results for debugging and verification.
            - At the end, return a concise textual summary of:
            - What you did
            - Key intermediate results
            - Any important caveats or assumptions

            Return only the answer (no explanation).
            """

            if sid == last_id:
                executor_prompt += f"""
                IMPORTANT: This is the FINAL subtask. After completing it, you MUST write the final answer and complete code pipeline to files using the write_file tool.
                - Write your final answer to {answer_path}
                - Write your complete code pipeline to {pipeline_code_path}
                You can end the task after writing the files.
                """
            subtask_output = executor_agent.run(executor_prompt)
            subtask_outputs[sid] = subtask_output
            agent_token_counts = self._get_token_counts(task_id)
            token_counts["input_tokens"] += agent_token_counts["input_tokens"]
            token_counts["output_tokens"] += agent_token_counts["output_tokens"]

        # Define final answer as the output of the last subtask (by id order)
        final_answer = subtask_outputs[last_id]

        return {
            "subtasks": subtasks,
            "subtask_outputs": subtask_outputs,
            "final_answer": final_answer,
            "token_usage": token_counts["input_tokens"] + token_counts["output_tokens"],
            "token_usage_input": token_counts["input_tokens"],
            "token_usage_output": token_counts["output_tokens"],
        }

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
    def serve_query(self, query: str, query_id: str = "default_name-0", subset_files: List[str] = []) -> Dict:
        """
        Serve a query using the LLM with a hard 15-minute timeout.

        If the PDT pipeline exceeds 15 minutes (900 seconds),
        the process is terminated and a timeout response is returned.
        """
        TIME_LIMIT = 1200  # 20 minutes

        # Prepare output directory
        self._init_output_dir(query_id)
        dataset_name = query_id.split("-")[0]
        dataset_directory = os.path.join(os.getcwd(), "data", dataset_name, "input")

        # -------------------------------------------------------
        # Worker function that runs the PDT pipeline
        # -------------------------------------------------------
        result_container = {"result": None, "error": None}

        def worker():
            try:
                planner_agent, decomposer_agent, executor_agent = self.create_pdt_agents(query_id)

                task_prompt = f"""
                Workload name: {dataset_name}
                dataset_directory = {dataset_directory}
                Answer the question: {query}; Use the {dataset_name} dataset.

                You are part of a Planner → Subtask Decomposer → Tool/Code Executor (PDT) architecture.
                Your role will be determined by the orchestrator calling you (planner_agent, decomposer_agent, executor_agent).
                """

                pdt_result = self.run_pdt_pipeline(
                    planner_agent,
                    decomposer_agent,
                    executor_agent,
                    task_prompt,
                    dataset_name,
                    query_id
                )

                result_container["result"] = pdt_result

            except Exception as e:
                result_container["error"] = str(e)

        # -------------------------------------------------------
        # Run the worker thread with a hard timeout
        # -------------------------------------------------------
        start_time = time.time()
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=TIME_LIMIT)
        runtime = time.time() - start_time

        # -------------------------------------------------------
        # Check for timeout
        # -------------------------------------------------------
        if thread.is_alive():
            # Thread timed out
            return {
                "id": query_id,
                "runtime": runtime,
                "timeout": True,
                "message": f"Query exceeded the 20-minute time limit (1200 seconds).",
                "subtasks": None,
                "subtask_outputs": None,
                "explanation": {"id": "main-task", "answer": None},
                "token_usage": 0,
                "token_usage_input": 0,
                "token_usage_output": 0,
                "pipeline_code": "N/A",
            }

        # -------------------------------------------------------
        # Pipeline finished normally
        # -------------------------------------------------------
        if result_container["error"] is not None:
            return {
                "id": query_id,
                "runtime": runtime,
                "timeout": False,
                "error": result_container["error"],
                "subtasks": None,
                "subtask_outputs": None,
                "explanation": {"id": "main-task", "answer": None},
                "token_usage": 0,
                "token_usage_input": 0,
                "token_usage_output": 0,
                "pipeline_code": "N/A",
            }

        pdt_result = result_container["result"]
        answer_path = os.path.join(self.question_intermediate_dir, f"answer.txt")
        pipeline_code_path = os.path.join(self.question_intermediate_dir, f"pipeline_code.py")
        answer, pipeline_code = self._get_output(answer_path, pipeline_code_path)
        

        results = {
            "id": query_id,
            "runtime": runtime,
            "timeout": False,
            "subtasks": pdt_result["subtasks"],
            "subtask_outputs": pdt_result["subtask_outputs"],
            "explanation": {"id": "main-task", "answer": pdt_result["final_answer"]},
            "token_usage": pdt_result["token_usage"],
            "token_usage_input": pdt_result["token_usage_input"],
            "token_usage_output": pdt_result["token_usage_output"],
            "pipeline_code": pipeline_code,
        }

        #print(results)
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
    out_dir = os.path.join(current_dir, "../../testresults/pdt")
    smolagents_dr = SmolagentsPDT(
        model="claude-3-7-sonnet-latest", output_dir=out_dir, verbose=True
    )
    for question in questions:
        print(f"Processing question: {question['id']}")
        # For debugging purposes, also input question.id
        smolagents_dr.serve_query(question["query"], question["id"])
        break

if __name__ == "__main__":
    main()