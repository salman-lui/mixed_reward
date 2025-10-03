#!/usr/bin/env python3
"""Merge original verifications with critiques using DeepCritic methodology"""

import sys
import os
import time
import json
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

NUM_VERIFICATIONS = 1
TEMPERATURE = 0.0
MAX_RETRIES = 10
MAX_TOKENS = 8192
TOP_P = 0.95

BATCH_SIZE = 200

DEBUG = False
DEBUG_PROBLEMS = 1

DATA_DIR = "/code/salsrahm-sandbox/data-reasoning/data-skymath/final_verification_data/without_reference/step_level/verification_final_match_step_level_n_16.jsonl"

VERIFICATION_FIELD_NAME = os.path.splitext(os.path.basename(DATA_DIR))[0]
CRITIQUE_FIELD_NAME = f"critiques_of_{VERIFICATION_FIELD_NAME}"
OUTPUT_DIR = f"/code/salsrahm-sandbox/data-reasoning/data-skymath/final_verification_data/without_reference/step_level/{VERIFICATION_FIELD_NAME}_merged.jsonl"

print(f"Auto-extracted configuration:")
print(f"   VERIFICATION_FIELD_NAME: {VERIFICATION_FIELD_NAME}")
print(f"   CRITIQUE_FIELD_NAME: {CRITIQUE_FIELD_NAME}")
print(f"   OUTPUT_FILE: {os.path.basename(OUTPUT_DIR)}")
print()

DATASET_TYPE = "VERIFICATION"

VERIFICATION_PROMPT = r"""You are a math expert and a good math critic.

You will be provided with an original verification and a critique of that verification.

Your task is to merge the two into a single, improved verification that incorporates the insights from the critique.

You should merge them as if they were generated in one go, as if the verifier first generated a verification and then wanted to further verify and improve their analysis.

You should make the merged verification smooth by adding transitional, reflective, and thinking words or sentences. Do not use terms like "the original verification" or "the critique says" as the merged verification should be considered as generated in one go.

If the critique identified any errors in the original verification's analysis:
- Correct those errors in the merged verification
- Provide the accurate mathematical reasoning  
- Update step labels (correct/incorrect) if needed
- Ensure the final verdict matches the corrected analysis

If the critique confirmed the original verification was accurate:
- Keep the original analysis but enhance it with additional insights as suggested by the critique
- Add more thorough explanations where beneficial
- Maintain the same step labels and final verdict

The output must follow the EXACT same format as the original verification:
- Start with "Teacher Verification:" (if present in original)
- Use "## Step X:" headers for each step analysis
- End each step with "**This step is correct.**" or "**This step is incorrect.**"
- Conclude with "**Verification: Is the answer correct (Yes/No)? X**"
- Do NOT add any additional sections, reflections, or commentary beyond this format

<Problem> __QUESTION_PLACEHOLDER__ </Problem>
<Solution Path> __SOLUTION_PLACEHOLDER__ </Solution Path>
<Original Verification> __ORIGINAL_VERIFICATION_PLACEHOLDER__ </Original Verification>
<Critique of the Verification> __CRITIQUE_PLACEHOLDER__ </Critique of the Verification>

Generate the merged verification in the exact format shown above:
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
ENDPOINT = "http://salsrahm-prm-1753207711-router.default.svc.cluster.local:8000/v1"

file_lock = Lock()

def separate_think_answer(verification_text):
    """Separate verification text and return only answer part after </think> tag"""
    if not verification_text or not verification_text.strip():
        return verification_text
    
    if not verification_text.strip().startswith('<think>'):
        return verification_text.strip()
    
    think_end_match = re.search(r'</think>', verification_text, re.IGNORECASE)
    if think_end_match:
        think_end_pos = think_end_match.end()
        answer_part = verification_text[think_end_pos:].strip()
        return answer_part
    
    step_match = re.search(r'\n## Step 1', verification_text)
    if step_match:
        think_end_pos = step_match.start()
        answer_part = verification_text[think_end_pos:].strip()
        return answer_part
    
    return verification_text.strip()

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
    """Load solutions with verifications and critiques from JSONL file"""
    print(f"Loading solutions with verifications and critiques from: {DATA_DIR}")
    
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
    """Process a single solution and generate merged verifications"""
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
        
        if VERIFICATION_FIELD_NAME not in solution_data or not solution_data[VERIFICATION_FIELD_NAME]:
            print(f"   Error: No {VERIFICATION_FIELD_NAME} found for Solution {solution_idx + 1}")
            return {"success": False, "error": f"No {VERIFICATION_FIELD_NAME} found", "solution_idx": solution_idx}
        
        verification_obj = solution_data[VERIFICATION_FIELD_NAME]
        if 'verification_text' not in verification_obj or not verification_obj['verification_text']:
            print(f"   Error: Empty verification_text in {VERIFICATION_FIELD_NAME} for Solution {solution_idx + 1}")
            return {"success": False, "error": "Empty verification text", "solution_idx": solution_idx}
        
        original_verification = verification_obj['verification_text']
        
        if CRITIQUE_FIELD_NAME not in solution_data or not solution_data[CRITIQUE_FIELD_NAME]:
            print(f"   Error: No {CRITIQUE_FIELD_NAME} found for Solution {solution_idx + 1}")
            return {"success": False, "error": f"No {CRITIQUE_FIELD_NAME} found", "solution_idx": solution_idx}
        
        critique_array = solution_data[CRITIQUE_FIELD_NAME]
        if not critique_array or len(critique_array) == 0:
            print(f"   Error: Empty {CRITIQUE_FIELD_NAME} array for Solution {solution_idx + 1}")
            return {"success": False, "error": "Empty critique array", "solution_idx": solution_idx}
        
        critique_obj = critique_array[0]
        if 'critique_text' not in critique_obj or not critique_obj['critique_text']:
            print(f"   Error: Empty critique_text in {CRITIQUE_FIELD_NAME} for Solution {solution_idx + 1}")
            return {"success": False, "error": "Empty critique text", "solution_idx": solution_idx}
        
        critique_text = critique_obj['critique_text']
        
        verification_answer_only = separate_think_answer(original_verification)
        critique_answer_only = separate_think_answer(critique_text)
        
        print(f"   Original verification length: {len(original_verification)} chars")
        print(f"   After removing <think>: {len(verification_answer_only)} chars")
        print(f"   Critique length: {len(critique_text)} chars")
        print(f"   Critique after removing <think>: {len(critique_answer_only)} chars")
        
        prompt = PROMPT_TEMPLATE.replace('__QUESTION_PLACEHOLDER__', question)
        prompt = prompt.replace('__SOLUTION_PLACEHOLDER__', solution)
        prompt = prompt.replace('__ORIGINAL_VERIFICATION_PLACEHOLDER__', verification_answer_only)
        prompt = prompt.replace('__CRITIQUE_PLACEHOLDER__', critique_answer_only)
        messages = [{"role": "user", "content": prompt}]
        
        verifications = []
        
        for verification_idx in range(NUM_VERIFICATIONS):
            try:
                print(f"   Generating merged verification {verification_idx + 1}/{NUM_VERIFICATIONS} for Solution {solution_idx + 1}...")
                
                response = agent.call_api(
                    messages=messages,
                    temperature=TEMPERATURE
                )
                
                verifications.append({
                    "merged_verification_id": verification_idx + 1,
                    "merged_verification_text": response.strip(),
                    "generated_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"   Error generating merged verification {verification_idx + 1} for Solution {solution_idx + 1}: {e}")
                verifications.append({
                    "merged_verification_id": verification_idx + 1,
                    "merged_verification_text": None,
                    "error": str(e),
                    "generated_at": datetime.now().isoformat()
                })
        
        output_record = solution_data.copy()
        output_record[f"merged_{VERIFICATION_FIELD_NAME}"] = verifications
        output_record["num_merged_verifications_generated"] = len([v for v in verifications if v["merged_verification_text"] is not None])
        output_record["merge_config"] = {
            "model": MODEL_PATH,
            "temperature": TEMPERATURE,
            "num_merged_verifications": NUM_VERIFICATIONS,
            "provider": PROVIDER,
            "dataset_type": DATASET_TYPE,
            "original_verification_field": VERIFICATION_FIELD_NAME,
            "critique_field": CRITIQUE_FIELD_NAME,
            "stage": "final_critique_synthesis",
            "think_section_removed": True,
            "original_verification_length": len(original_verification),
            "processed_verification_length": len(verification_answer_only),
            "critique_length": len(critique_text),
            "processed_critique_length": len(critique_answer_only)
        }
        
        if save_result_thread_safe(output_record, output_file):
            print(f"Solution {solution_idx + 1}: Saved {output_record['num_merged_verifications_generated']}/{NUM_VERIFICATIONS} merged verifications")
            return {"success": True, "solution_idx": solution_idx, "num_merged": output_record['num_merged_verifications_generated']}
        else:
            print(f"Solution {solution_idx + 1}: Failed to save merged verifications")
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

def generate_merged_verifications():
    """Generate merged verifications for all solutions using parallel processing"""
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
    
    print(f"\nGenerating {NUM_VERIFICATIONS} merged verifications from {VERIFICATION_FIELD_NAME} + {CRITIQUE_FIELD_NAME} for {len(solutions)} solutions")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    start_time = time.time()
    print(f"Starting merged verification generation at: {time.strftime('%H:%M:%S')}")
    
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
        print(f"   Merged verifications generated so far: {total_successful * NUM_VERIFICATIONS}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nMERGED VERIFICATION GENERATION COMPLETE!")
    if DEBUG:
        print(f"DEBUG MODE: Only processed {len(solutions)} solutions for testing")
    print("=" * 80)
    print(f"Started: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print(f"Finished: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Solutions processed: {len(solutions)}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Total merged verifications generated: {total_successful * NUM_VERIFICATIONS}")
    print(f"Average speed: {len(solutions)/(total_time/60):.1f} solutions/minute")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_merged_verifications()