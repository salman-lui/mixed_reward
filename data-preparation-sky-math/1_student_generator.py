#!/usr/bin/env python3
"""
Solution Generator with Parallel Processing

Generates multiple solutions per problem using LLMs with parallel batch processing.
Supports MATH, NATURAL_REASONING, SKYWORK, and custom datasets via configurable field mappings.
"""

import sys
import os
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add the parent directory to the path to import base_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

# Configuration
NUM_SOLUTIONS = 20
TEMPERATURE = 0.7
MAX_RETRIES = 10
MAX_TOKENS = 8192
TOP_P = 0.95

BATCH_PROBLEM = 600

DEBUG = False
DEBUG_PROBLEMS = 15

DATA_DIR = "/code/salsrahm-sandbox/data-reasoning/data-skymath/eval_skywork.jsonl"
OUTPUT_DIR = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/skywork_solutions_generated_eval.jsonl"

DATASET_TYPE = "SKYWORK"
DATASET_CONFIGS = {
   "MATH": {
       "question_field": "problem",
       "answer_field": "solution",
       "metadata_fields": ["level", "type"],
       "prompt_template": """Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

Problem: {question}

Break down your solution into clear, numbered steps (Step 1, Step 2, etc.).
Explain your reasoning for each step.

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.

Solution:
"""
   },
  
   "NATURAL_REASONING": {
       "question_field": "question",
       "answer_field": "reference_answer",
       "metadata_fields": [],
       "prompt_template": """Answer this question step by step:

{question}

Use numbered steps (Step 1, Step 2, etc.) and explain your reasoning for each step.
"""
   },
  
   "SKYWORK": {
       "question_field": "prompt",
       "answer_field": "reward_model",
       "metadata_fields": ["data_source", "ability", "reward_model", "extra_info"],
       "prompt_template": """Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

Problem: {question}

Break down your solution into clear, numbered steps (Step 1, Step 2, etc.).
Explain your reasoning for each step.

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.

Solution:
"""
   },
  
   "CUSTOM": {
       "question_field": "question",
       "answer_field": "answer",
       "metadata_fields": ["category", "difficulty"],
       "prompt_template": """Answer the following question:

Question: {question}

Answer:
"""
   }
}

CURRENT_CONFIG = DATASET_CONFIGS[DATASET_TYPE]
QUESTION_FIELD = CURRENT_CONFIG["question_field"]
ANSWER_FIELD = CURRENT_CONFIG["answer_field"] 
METADATA_FIELDS = CURRENT_CONFIG["metadata_fields"]
PROMPT_TEMPLATE = CURRENT_CONFIG["prompt_template"]

PROVIDER = "kubernetes"
MODEL_PATH = "/checkpoints/salsrahm-sandbox/qwen-2.5-14b-instruct"
ENDPOINT = "http://salsrahm-qwen14-1752894955-router.default.svc.cluster.local:8000/v1"

file_lock = Lock()

def setup_agent():
    """Setup BaseAgent with the configured model and endpoint."""
    config = {
        "provider": PROVIDER,
        "model_path": MODEL_PATH,
        "endpoint": ENDPOINT,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "max_retries": MAX_RETRIES,
        "temperature": TEMPERATURE
    }
    
    # Display setup info
    if "localhost" in ENDPOINT:
        model_display = f"{MODEL_PATH} via SGLang Local ({ENDPOINT})"
    else:
        model_display = f"{MODEL_PATH} via Remote K8s ({ENDPOINT})"
    
    try:
        agent = BaseAgent(config)
        print(f"Agent initialized: {model_display}")
        return agent
    except Exception as e:
        print(f"Error setting up agent: {e}")
        return None

def load_problems():
    """Load problems from JSONL file."""
    print(f"Loading problems from: {DATA_DIR}")
    
    problems = []
    try:
        with open(DATA_DIR, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    problem_data = json.loads(line.strip())
                    problems.append(problem_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(problems)} problems successfully")
        return problems
    
    except FileNotFoundError:
        print(f"Error: File {DATA_DIR} not found")
        return []
    except Exception as e:
        print(f"Error loading problems: {e}")
        return []

def save_result_thread_safe(output_record, output_file):
    """Thread-safe save function for parallel processing."""
    try:
        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"Error saving result: {e}")
        return False

def process_single_problem(problem_data, problem_idx, output_file):
    """Process a single problem and generate solutions."""
    try:
        agent = setup_agent()
        if not agent:
            return {"success": False, "error": "Failed to setup agent", "problem_idx": problem_idx}
        
        metadata_info = []
        for field in METADATA_FIELDS:
            if field in problem_data:
                metadata_info.append(f"{field}: {problem_data[field]}")
        metadata_str = ", ".join(metadata_info) if metadata_info else "No metadata"
        
        print(f"Processing Problem {problem_idx + 1} (Thread ID: {problem_idx})")
        print(f"   {metadata_str}")
        
        if DATASET_TYPE == "SKYWORK":
            problem_text = problem_data[QUESTION_FIELD][0]["content"]
        else:
            problem_text = problem_data[QUESTION_FIELD]
        
        prompt = PROMPT_TEMPLATE.format(question=problem_text)
        messages = [{"role": "user", "content": prompt}]
        
        solutions = []
        
        for solution_idx in range(NUM_SOLUTIONS):
            try:
                print(f"   Generating solution {solution_idx + 1}/{NUM_SOLUTIONS} for Problem {problem_idx + 1}...")
                
                response = agent.call_api(
                    messages=messages,
                    temperature=TEMPERATURE
                )
                
                solutions.append({
                    "solution_id": solution_idx + 1,
                    "generated_solution": response.strip(),
                    "generated_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"   Error generating solution {solution_idx + 1} for Problem {problem_idx + 1}: {e}")
                solutions.append({
                    "solution_id": solution_idx + 1,
                    "generated_solution": None,
                    "error": str(e),
                    "generated_at": datetime.now().isoformat()
                })
        
        if DATASET_TYPE == "SKYWORK":
            original_answer = problem_data[ANSWER_FIELD]["ground_truth"]
        else:
            original_answer = problem_data[ANSWER_FIELD]
        
        output_record = {
            "problem_id": problem_idx + 1,
            "original_question": problem_text,
            "original_answer": original_answer,
            "generated_solutions": solutions,
            "num_solutions_generated": len([s for s in solutions if s["generated_solution"] is not None]),
            "generation_config": {
                "model": MODEL_PATH,
                "temperature": TEMPERATURE,
                "num_solutions": NUM_SOLUTIONS,
                "provider": PROVIDER,
                "dataset_type": DATASET_TYPE
            }
        }
        
        for field in METADATA_FIELDS:
            if field in problem_data:
                output_record[field] = problem_data[field]
        
        if save_result_thread_safe(output_record, output_file):
            print(f"Problem {problem_idx + 1}: Saved {output_record['num_solutions_generated']}/{NUM_SOLUTIONS} solutions")
            return {"success": True, "problem_idx": problem_idx, "num_solutions": output_record['num_solutions_generated']}
        else:
            print(f"Problem {problem_idx + 1}: Failed to save solutions")
            return {"success": False, "error": "Failed to save", "problem_idx": problem_idx}
            
    except Exception as e:
        print(f"Critical error processing problem {problem_idx + 1}: {e}")
        return {"success": False, "error": str(e), "problem_idx": problem_idx}

def process_batch_parallel(batch_problems, output_file, max_workers=None):
    """Process a batch of problems in parallel."""
    problem_ids = [original_idx + 1 for original_idx, _ in batch_problems]
    print(f"\nProcessing batch: Problems {min(problem_ids)} to {max(problem_ids)}")
    print(f"   Batch size: {len(batch_problems)} problems")
    print(f"   Max workers: {max_workers or 'auto'}")
    
    batch_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_problem = {
            executor.submit(process_single_problem, problem_data, original_idx, output_file): original_idx
            for original_idx, problem_data in batch_problems
        }
        
        for future in as_completed(future_to_problem):
            original_idx = future_to_problem[future]
            try:
                result = future.result()
                batch_results.append(result)
                
                if result["success"]:
                    print(f"   Completed Problem {original_idx + 1}")
                else:
                    print(f"   Failed Problem {original_idx + 1}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   Exception in Problem {original_idx + 1}: {e}")
                batch_results.append({"success": False, "error": str(e), "problem_idx": original_idx})
    
    batch_time = time.time() - start_time
    successful_in_batch = sum(1 for r in batch_results if r["success"])
    failed_in_batch = len(batch_results) - successful_in_batch
    
    print(f"   Batch completed in {batch_time:.1f}s")
    print(f"   Successful: {successful_in_batch}/{len(batch_problems)}")
    print(f"   Failed: {failed_in_batch}/{len(batch_problems)}")
    
    return batch_results

def get_completed_problem_ids(output_file):
    """Get list of already completed problem IDs from output file."""
    if not os.path.exists(output_file):
        return set()
    
    completed_ids = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'problem_id' in record:
                        completed_ids.add(record['problem_id'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading existing output file: {e}")
        return set()
    
    return completed_ids

def generate_solutions_for_all_problems():
    """Generate solutions for all problems in the dataset."""
    global OUTPUT_DIR
    
    problems = load_problems()
    if not problems:
        print("No problems loaded. Exiting.")
        return
    
    if DEBUG:
        original_count = len(problems)
        problems = problems[:DEBUG_PROBLEMS]
        print(f"DEBUG MODE: Processing only first {len(problems)} problems out of {original_count}")
        
        base_name = OUTPUT_DIR.replace('.jsonl', '_debug.jsonl')
        OUTPUT_DIR = base_name
        print(f"DEBUG OUTPUT: {OUTPUT_DIR}")
    
    completed_ids = get_completed_problem_ids(OUTPUT_DIR)
    if completed_ids:
        print(f"RESUME MODE: Found {len(completed_ids)} already completed problems")
        print(f"   Skipping problems: {sorted(list(completed_ids))}")
        
        problems_to_process = []
        for idx, problem in enumerate(problems):
            problem_id = idx + 1
            if problem_id not in completed_ids:
                problems_to_process.append((idx, problem))
        
        print(f"   Remaining problems to process: {len(problems_to_process)}")
        
        if not problems_to_process:
            print("All problems already completed.")
            return
            
        problems = problems_to_process
    else:
        print("FRESH START: No existing output file found")
        problems = [(idx, problem) for idx, problem in enumerate(problems)]
    
    effective_batch_size = min(BATCH_PROBLEM, len(problems)) if not DEBUG else len(problems)
    max_workers = effective_batch_size
    
    print(f"Parallel processing configuration:")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Max workers per batch: {max_workers}")
    
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    
    if not completed_ids and os.path.exists(OUTPUT_DIR):
        print(f"Clearing existing output file: {OUTPUT_DIR}")
        open(OUTPUT_DIR, 'w').close()
    
    print(f"\nGenerating {NUM_SOLUTIONS} solutions for {len(problems)} problems")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    start_time = time.time()
    print(f"Starting generation at: {time.strftime('%H:%M:%S')}")
    
    total_successful = 0
    total_failed = 0
    all_results = []
    
    for batch_start in range(0, len(problems), effective_batch_size):
        batch_end = min(batch_start + effective_batch_size, len(problems))
        batch_problems = problems[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_start//effective_batch_size + 1}/{(len(problems)-1)//effective_batch_size + 1}")
        print(f"{'='*60}")
        
        batch_results = process_batch_parallel(
            batch_problems, 
            OUTPUT_DIR, 
            max_workers=max_workers
        )
        
        batch_successful = sum(1 for r in batch_results if r["success"])
        batch_failed = len(batch_results) - batch_successful
        
        total_successful += batch_successful
        total_failed += batch_failed
        all_results.extend(batch_results)
        
        problems_processed = batch_end
        elapsed = time.time() - start_time
        problems_per_minute = problems_processed / (elapsed / 60) if elapsed > 0 else 0
        remaining_problems = len(problems) - problems_processed
        eta_minutes = remaining_problems / problems_per_minute if problems_per_minute > 0 else 0
        
        print(f"\nProgress:")
        print(f"   {problems_processed}/{len(problems)} ({problems_processed/len(problems)*100:.1f}%)")
        print(f"   {problems_per_minute:.1f} problems/minute")
        print(f"   {eta_minutes:.1f} minutes remaining")
        print(f"   Successful: {total_successful}")
        print(f"   Failed: {total_failed}")
        print(f"   Solutions: {total_successful * NUM_SOLUTIONS}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nGeneration complete")
    if DEBUG:
        print(f"DEBUG: Processed {len(problems)} problems")
    print("=" * 80)
    print(f"Started: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print(f"Finished: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Problems: {len(problems)}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Solutions: {total_successful * NUM_SOLUTIONS}")
    print(f"Speed: {len(problems)/(total_time/60):.1f} problems/minute")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_solutions_for_all_problems()