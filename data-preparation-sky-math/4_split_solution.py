#!/usr/bin/env python3
"""Flatten nested solutions structure so each solution is on its own line"""

import json
import sys
import os

INPUT_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/skywork_solutions_generated_eval_with_verification_reduced_balanced.jsonl"
OUTPUT_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/skywork_solutions_generated_eval_with_verification_reduced_balanced_split_by_solution.jsonl"

def split_solutions():
   """Split nested balanced solutions into one solution per line"""
   print(f"Loading nested solutions from: {INPUT_FILE}")
  
   if not os.path.exists(INPUT_FILE):
       print(f"Error: Input file {INPUT_FILE} not found!")
       return
  
   with open(OUTPUT_FILE, 'w') as f:
       pass
  
   total_problems = 0
   total_solutions = 0
  
   try:
       with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
           with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
              
               for line_num, line in enumerate(infile, 1):
                   try:
                       problem_data = json.loads(line.strip())
                      
                       problem_id = problem_data.get("problem_id", f"problem_{line_num}")
                       original_question = problem_data["original_question"]
                       original_answer = problem_data.get("original_answer", "")
                       data_source = problem_data.get("data_source", "")
                       ability = problem_data.get("ability", "")
                       correct_answer = problem_data.get("reward_model", {}).get("ground_truth", [])
                      
                       generated_solutions = problem_data.get("generated_solutions", [])
                      
                       print(f"Processing Problem {problem_id}: {len(generated_solutions)} solutions")
                      
                       for solution_data in generated_solutions:
                           solution_record = {
                               "problem_id": problem_id,
                               "original_question": original_question,
                               "original_answer": original_answer,
                               "data_source": data_source,
                               "ability": ability,
                               "correct_answer": correct_answer,
                               "solution": solution_data.get("generated_solution", ""),
                               "is_correct": solution_data.get("is_correct", ""),
                               "solution_id": solution_data.get("solution_id", 0),
                               "extracted_answer": solution_data.get("extracted_answer", ""),
                               "generated_at": solution_data.get("generated_at", "")
                           }
                          
                           outfile.write(json.dumps(solution_record, ensure_ascii=False) + '\n')
                           total_solutions += 1
                      
                       total_problems += 1
                      
                   except json.JSONDecodeError as e:
                       print(f"Error parsing line {line_num}: {e}")
                       continue
                   except KeyError as e:
                       print(f"Missing key in line {line_num}: {e}")
                       continue
                   except Exception as e:
                       print(f"Error processing line {line_num}: {e}")
                       continue
  
   except Exception as e:
       print(f"Error processing file: {e}")
       return
  
   print(f"\nSplit complete!")
   print(f"Problems processed: {total_problems}")
   print(f"Solutions split: {total_solutions}")
   print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
   split_solutions()

