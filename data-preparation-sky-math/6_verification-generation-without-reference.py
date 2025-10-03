#!/usr/bin/env python3
"""Verification generator without ground truth access for reference-free training"""

import sys
import os
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

NUM_VERIFICATIONS = 16
TEMPERATURE = 0.8
MAX_RETRIES = 200
MAX_TOKENS = 8192
TOP_P = 0.95

BATCH_SIZE = 1300

DEBUG = False
DEBUG_PROBLEMS = 5

DATA_DIR = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/without_reference_verification/remaining-data-verification/skywork_solutions_remaining_train2.jsonl"
OUTPUT_DIR = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/without_reference_verification/part-1-done-verification/skywork_verifications_no_reference_train_part_4.jsonl"

DATASET_TYPE = "VERIFICATION"


VERIFICATION_PROMPT = r"""You are a math teacher grading student work. Your task is to verify the student's solution step by step and identify any errors. For each step, provide your analysis followed by a verdict. You must check the mathematical reasoning, calculations, and logic independently for each step.

Important: Each step must be marked as either "correct" or "incorrect" - no partial credit. If a step has any errors or omissions, mark it as incorrect.

At the end of your verification, provide your final verdict in the form: 'Verification: Is the answer correct (Yes/No)? X', where X is either Yes or No.

Example 1:

Question: 
Sam is hired for a 20-day period. On days that he works, he earns $\$$60. For each day that he does not work, $\$$30 is subtracted from his earnings. At the end of the 20-day period, he received $\$$660. How many days did he not work?

Student Solution: 

Step 1: Define the variable 
Let $n$ be the number of days that he did not work. Then $20-n$ is the number of days that he did work. 

Step 2: Set up the earnings equation
His earnings for working for $20-n$ days is $60(20-n)$, and his earnings for not working for $n$ days is $-30n$, since $30 is subtracted for each day that he did not work. 

Step 3: Write and solve the equation
\begin{align*}
60(20-n) - 30n &= 660\\
\Rightarrow\qquad 1200-60n-30n &= 660\\
\Rightarrow\qquad -90n &= -540\\
\Rightarrow\qquad n &= \frac{-540}{-90} = \boxed{6}.
\end{align*}

Final Answer: The final answer is $6$.

Teacher Verification: 

## Step 1: Define the variable
The student used $n$ for days not worked and $(20-n)$ for days worked. This correctly captures that the total days must sum to 20, which is the correct approach for this problem. 

**This step is correct.**

## Step 2: Set up the earnings equation
The earnings calculation properly accounts for both components: $60(20-n)$ for the money earned from working days and $-30n$ for the deductions from non-working days. Setting this equal to $660$ gives the correct equation to solve the problem. 

**This step is correct.**

## Step 3: Write and solve the equation
Let me check each algebraic step. The expansion of $60(20-n)$ gives $1200 - 60n$, which is correct. Combining $-60n - 30n$ yields $-90n$, also correct. This produces the equation $1200 - 90n = 660$. Solving this equation: subtracting 1200 from both sides gives $-90n = -540$, and dividing by -90 gives $n = 6$. All algebraic manipulations are performed correctly.

**This step is correct.**

**Verification: Is the answer correct (Yes/No)? Yes**
---

Example 2:

Question: 
Tim wants to invest some money in a bank which compounds quarterly with an annual interest rate of $7\%$. To the nearest dollar, how much money should he invest if he wants a total of $\$60,\!000$ at the end of $5$ years?

Student Solution:
Step 1: Understand the problem and the formula for compound interest
The formula for the amount $A$ in an account after $n$ years, with principal $P$, annual interest rate $r$ (in decimal), and compounding $q$ times per year is given by $A = P\left(1 + \frac{r}{q}\right)^{qn}$. Here, Tim wants to invest an amount $P$ which will become $60000$ after $5$ years with an annual interest rate of $7\%$ compounded quarterly. So we have $A = 60000$, $r = 0.07$, $q = 4$, and $n = 5$.

Step 2: Plug the given values into the formula
We plug $A = 60000$, $r = 0.07$, $q = 4$, and $n = 5$ into the formula $A = P\left(1 + \frac{r}{q}\right)^{qn}$ to get $60000 = P\left(1 + \frac{0.07}{4}\right)^{4 \cdot 5}$.

Step 3: Solve for P
Solving the equation for $P$, we have $P = \frac{60000}{\left(1 + \frac{0.07}{4}\right)^{20}}$.

Step 4: Calculate the value
Let's calculate the value of $P$. $P = \frac{60000}{\left(1 + \frac{0.07}{4}\right)^{20}} \approx \frac{60000}{\left(1 + 0.0175\right)^{20}} \approx \frac{60000}{\left(1.0175\right)^{20}} \approx \frac{60000}{1.419067227}$.

Step 5: Compute the final value
Computing the final value, $P \approx \frac{60000}{1.419067227} \approx 42353.91884$.

Step 6: Round the answer to the nearest dollar
Rounding to the nearest dollar, $P \approx 42354$.

The final answer is: $42354$

Teacher Verification:

## Step 1: Understand the problem and the formula for compound interest
The student correctly identified the compound interest formula and properly defined all variables. The values $A = 60000$, $r = 0.07$, $q = 4$, and $n = 5$ are all correct for this problem. 

**This step is correct.**

## Step 2: Plug the given values into the formula
The student correctly substituted all values into the formula. With $A = 60000$, $r = 0.07$, $q = 4$, and $n = 5$, the equation becomes $60000 = P\left(1 + \frac{0.07}{4}\right)^{4 \cdot 5}$. 

**This step is correct.**

## Step 3: Solve for P
The rearrangement to solve for $P$ is correct: $P = \frac{60000}{\left(1 + \frac{0.07}{4}\right)^{20}}$. 

**This step is correct.**

## Step 4: Calculate the value
The student calculated $1 + \frac{0.07}{4} = 1.0175$ correctly. However, when computing $(1.0175)^{20}$, the student got approximately $1.419067227$. Let me verify this calculation. Actually, $(1.0175)^{20} \approx 1.41478...$, not $1.419067227$. This is a calculation error.

**This step is incorrect.**

## Step 5: Compute the final value
Because the previous calculation was incorrect, this division $\frac{60000}{1.419067227} \approx 42353.91884$ produces an incorrect result. With the correct value of $(1.0175)^{20} \approx 1.41478$, we should get $P \approx \frac{60000}{1.41478} \approx 42409.47$. The error from Step 4 has propagated to this step.

**This step is incorrect.**

## Step 6: Round the answer to the nearest dollar
The student correctly rounded their calculated value, but since the value itself was incorrect, the final answer of $42354$ is wrong. The correct answer should be $42409$. Although the rounding procedure is correct, the input value is wrong.

**This step is incorrect.**

**Verification: Is the answer correct (Yes/No)? No**

---

Now, grade the following student solution step by step as follows:

Question: __QUESTION_PLACEHOLDER__ 

Student Solution: __SOLUTION_PLACEHOLDER__

Teacher Verification:
"""

DATASET_CONFIGS = {
    "VERIFICATION": {
        "question_field": "original_question",
        "answer_field": "original_answer",
        "solution_field": "solution",
        "metadata_fields": ["data_source", "ability", "problem_id", "solution_id", "is_correct", "extracted_answer", "generated_at"],
        "prompt_template": VERIFICATION_PROMPT
    }
}

CURRENT_CONFIG = DATASET_CONFIGS[DATASET_TYPE]
QUESTION_FIELD = CURRENT_CONFIG["question_field"]
ANSWER_FIELD = CURRENT_CONFIG["answer_field"] 
SOLUTION_FIELD = CURRENT_CONFIG["solution_field"]
METADATA_FIELDS = CURRENT_CONFIG["metadata_fields"]
PROMPT_TEMPLATE = CURRENT_CONFIG["prompt_template"]

PROVIDER = "kubernetes"
MODEL_PATH = "/checkpoints/salsrahm-sandbox/qwen3-32b"
ENDPOINT = "http://salsrahm-ag3-9eval-1753001416-router.default.svc.cluster.local:8000/v1"

file_lock = Lock()

def setup_agent():
    """Setup BaseAgent with configured model and endpoint"""
    config = {
        "provider": PROVIDER,
        "model_path": MODEL_PATH,
        "endpoint": ENDPOINT,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "max_retries": MAX_RETRIES,
        "temperature": TEMPERATURE
    }
    
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

def load_solutions():
    """Load split solutions from JSONL file"""
    print(f"Loading split solutions from: {DATA_DIR}")
    
    solutions = []
    try:
        with open(DATA_DIR, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    solution_data = json.loads(line.strip())
                    solutions.append(solution_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(solutions)} solutions successfully")
        return solutions
    
    except FileNotFoundError:
        print(f"Error: File {DATA_DIR} not found")
        return []
    except Exception as e:
        print(f"Error loading solutions: {e}")
        return []

def save_result_thread_safe(output_record, output_file):
    """Thread-safe save function for parallel processing"""
    try:
        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"Error saving result: {e}")
        return False

def process_single_solution(solution_data, solution_idx, output_file):
    """Process a single solution and generate verifications"""
    try:
        agent = setup_agent()
        if not agent:
            return {"success": False, "error": "Failed to setup agent", "solution_idx": solution_idx}
        
        metadata_info = []
        for field in METADATA_FIELDS:
            if field in solution_data:
                metadata_info.append(f"{field}: {solution_data[field]}")
        metadata_str = ", ".join(metadata_info) if metadata_info else "No metadata"
        
        print(f"Processing Solution {solution_idx + 1} (Thread ID: {solution_idx})")
        print(f"   {metadata_str}")
        
        question = solution_data[QUESTION_FIELD]
        solution = solution_data[SOLUTION_FIELD]
        
        prompt = PROMPT_TEMPLATE.replace('__QUESTION_PLACEHOLDER__', question)
        prompt = prompt.replace('__SOLUTION_PLACEHOLDER__', solution)
        messages = [{"role": "user", "content": prompt}]
        
        verifications = []
        
        for verification_idx in range(NUM_VERIFICATIONS):
            try:
                print(f"   Generating verification {verification_idx + 1}/{NUM_VERIFICATIONS} for Solution {solution_idx + 1}...")
                
                response = agent.call_api(
                    messages=messages,
                    temperature=TEMPERATURE
                )
                
                verifications.append({
                    "verification_id": verification_idx + 1,
                    "verification_text": response.strip(),
                    "generated_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"   Error generating verification {verification_idx + 1} for Solution {solution_idx + 1}: {e}")
                verifications.append({
                    "verification_id": verification_idx + 1,
                    "verification_text": None,
                    "error": str(e),
                    "generated_at": datetime.now().isoformat()
                })
        
        output_record = solution_data.copy()
        output_record["verifications"] = verifications
        output_record["num_verifications_generated"] = len([v for v in verifications if v["verification_text"] is not None])
        output_record["verification_config"] = {
            "model": MODEL_PATH,
            "temperature": TEMPERATURE,
            "num_verifications": NUM_VERIFICATIONS,
            "provider": PROVIDER,
            "dataset_type": DATASET_TYPE
        }
        
        if save_result_thread_safe(output_record, output_file):
            print(f"Solution {solution_idx + 1}: Saved {output_record['num_verifications_generated']}/{NUM_VERIFICATIONS} verifications")
            return {"success": True, "solution_idx": solution_idx, "num_verifications": output_record['num_verifications_generated']}
        else:
            print(f"Solution {solution_idx + 1}: Failed to save verifications")
            return {"success": False, "error": "Failed to save", "solution_idx": solution_idx}
            
    except Exception as e:
        print(f"Critical error processing solution {solution_idx + 1}: {e}")
        return {"success": False, "error": str(e), "solution_idx": solution_idx}

def process_batch_parallel(batch_solutions, output_file, max_workers=None):
    """Process a batch of solutions in parallel using ThreadPoolExecutor"""
    solution_ids = [original_idx + 1 for original_idx, _ in batch_solutions]
    print(f"\nProcessing batch: Solutions {min(solution_ids)} to {max(solution_ids)}")
    print(f"   Batch size: {len(batch_solutions)} solutions")
    print(f"   Max workers: {max_workers or 'auto'}")
    
    batch_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_solution = {
            executor.submit(process_single_solution, solution_data, original_idx, output_file): original_idx
            for original_idx, solution_data in batch_solutions
        }
        
        for future in as_completed(future_to_solution):
            original_idx = future_to_solution[future]
            try:
                result = future.result()
                batch_results.append(result)
                
                if result["success"]:
                    print(f"   Completed Solution {original_idx + 1}")
                else:
                    print(f"   Failed Solution {original_idx + 1}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   Exception in Solution {original_idx + 1}: {e}")
                batch_results.append({"success": False, "error": str(e), "solution_idx": original_idx})
    
    batch_time = time.time() - start_time
    successful_in_batch = sum(1 for r in batch_results if r["success"])
    failed_in_batch = len(batch_results) - successful_in_batch
    
    print(f"   Batch completed in {batch_time:.1f}s")
    print(f"   Successful: {successful_in_batch}/{len(batch_solutions)}")
    print(f"   Failed: {failed_in_batch}/{len(batch_solutions)}")
    
    return batch_results

def generate_verifications_for_all_solutions():
    """Generate verifications for all solutions using parallel processing"""
    global OUTPUT_DIR
    
    solutions = load_solutions()
    if not solutions:
        print("No solutions loaded. Exiting.")
        return
    
    if DEBUG:
        original_count = len(solutions)
        solutions = solutions[:DEBUG_PROBLEMS]
        print(f"DEBUG MODE: Processing only first {len(solutions)} solutions out of {original_count}")
        print("   To process full dataset, set DEBUG = False in configuration")
    
    solutions = [(idx, solution) for idx, solution in enumerate(solutions)]
    
    effective_batch_size = min(BATCH_SIZE, len(solutions)) if not DEBUG else len(solutions)
    max_workers = effective_batch_size
    
    print(f"Parallel processing configuration:")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Max workers per batch: {max_workers}")
    
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing existing output file: {OUTPUT_DIR}")
        open(OUTPUT_DIR, 'w').close()
    
    print(f"\nGenerating {NUM_VERIFICATIONS} verifications for {len(solutions)} solutions")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    start_time = time.time()
    print(f"Starting verification generation at: {time.strftime('%H:%M:%S')}")
    
    total_successful = 0
    total_failed = 0
    all_results = []
    
    for batch_start in range(0, len(solutions), effective_batch_size):
        batch_end = min(batch_start + effective_batch_size, len(solutions))
        batch_solutions = solutions[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_start//effective_batch_size + 1}/{(len(solutions)-1)//effective_batch_size + 1}")
        print(f"{'='*60}")
        
        batch_results = process_batch_parallel(
            batch_solutions, 
            OUTPUT_DIR, 
            max_workers=max_workers
        )
        
        batch_successful = sum(1 for r in batch_results if r["success"])
        batch_failed = len(batch_results) - batch_successful
        
        total_successful += batch_successful
        total_failed += batch_failed
        all_results.extend(batch_results)
        
        solutions_processed = batch_end
        elapsed = time.time() - start_time
        solutions_per_minute = solutions_processed / (elapsed / 60) if elapsed > 0 else 0
        remaining_solutions = len(solutions) - solutions_processed
        eta_minutes = remaining_solutions / solutions_per_minute if solutions_per_minute > 0 else 0
        
        print(f"\nOVERALL PROGRESS:")
        print(f"   Solutions processed: {solutions_processed}/{len(solutions)} ({solutions_processed/len(solutions)*100:.1f}%)")
        print(f"   Speed: {solutions_per_minute:.1f} solutions/minute")
        print(f"   ETA: {eta_minutes:.1f} minutes remaining")
        print(f"   Total successful: {total_successful}")
        print(f"   Total failed: {total_failed}")
        print(f"   Verifications generated so far: {total_successful * NUM_VERIFICATIONS}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nVERIFICATION GENERATION COMPLETE!")
    if DEBUG:
        print(f"DEBUG MODE: Only processed {len(solutions)} solutions for testing")
    print("=" * 80)
    print(f"Started: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print(f"Finished: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Solutions processed: {len(solutions)}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Total verifications generated: {total_successful * NUM_VERIFICATIONS}")
    print(f"Average speed: {len(solutions)/(total_time/60):.1f} solutions/minute")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_verifications_for_all_solutions()