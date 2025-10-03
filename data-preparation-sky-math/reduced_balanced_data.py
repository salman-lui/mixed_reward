#!/usr/bin/env python3
"""Reduce balanced dataset by capping solutions at 6 per problem (3 correct + 3 incorrect)"""

import json
import random
from collections import defaultdict

INPUT_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/skywork_solutions_generated_eval_with_verification_balanced.jsonl"
OUTPUT_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/skywork_solutions_generated_eval_with_verification_reduced_balanced.jsonl"
MAX_SOLUTIONS_PER_PROBLEM = 6
MAX_CORRECT_PER_PROBLEM = 3
MAX_INCORRECT_PER_PROBLEM = 3

def reduce_balanced_dataset():
   """Reduce balanced dataset by capping solutions at 6 per problem"""
  
   print(f"Reducing dataset from: {INPUT_FILE}")
   print(f"Max solutions per problem: {MAX_SOLUTIONS_PER_PROBLEM}")
   print(f"Max correct per problem: {MAX_CORRECT_PER_PROBLEM}")
   print(f"Max incorrect per problem: {MAX_INCORRECT_PER_PROBLEM}")
  
   original_stats = defaultdict(int)
   reduced_stats = defaultdict(int)
   total_problems = 0
   total_original_solutions = 0
   total_reduced_solutions = 0
   problems_reduced = 0
  
   reduced_data = []
  
   try:
       with open(INPUT_FILE, 'r', encoding='utf-8') as f:
           for line_num, line in enumerate(f, 1):
               try:
                   problem_data = json.loads(line.strip())
                  
                   generated_solutions = problem_data.get("generated_solutions", [])
                   num_solutions = len(generated_solutions)
                  
                   original_stats[num_solutions] += 1
                   total_problems += 1
                   total_original_solutions += num_solutions
                  
                   correct_solutions = []
                   incorrect_solutions = []
                  
                   for solution in generated_solutions:
                       if solution.get('is_correct') == 'correct':
                           correct_solutions.append(solution)
                       else:
                           incorrect_solutions.append(solution)
                  
                   if num_solutions <= MAX_SOLUTIONS_PER_PROBLEM:
                       final_solutions = generated_solutions
                       print(f"Problem {total_problems}: Keeping all {num_solutions} solutions")
                   else:
                       problems_reduced += 1
                      
                       random.shuffle(correct_solutions)
                       random.shuffle(incorrect_solutions)
                      
                       selected_correct = correct_solutions[:MAX_CORRECT_PER_PROBLEM]
                       selected_incorrect = incorrect_solutions[:MAX_INCORRECT_PER_PROBLEM]
                      
                       final_solutions = selected_correct + selected_incorrect
                      
                       print(f"Problem {total_problems}: Reduced {num_solutions} → {len(final_solutions)} solutions "
                             f"({len(correct_solutions)}→{len(selected_correct)} correct, "
                             f"{len(incorrect_solutions)}→{len(selected_incorrect)} incorrect)")
                  
                   final_count = len(final_solutions)
                   reduced_stats[final_count] += 1
                   total_reduced_solutions += final_count
                  
                   reduced_problem = problem_data.copy()
                   reduced_problem['generated_solutions'] = final_solutions
                   reduced_data.append(reduced_problem)
                  
               except json.JSONDecodeError as e:
                   print(f"Error parsing line {line_num}: {e}")
                   continue
               except Exception as e:
                   print(f"Error processing line {line_num}: {e}")
                   continue
  
   except Exception as e:
       print(f"Error reading file: {e}")
       return
  
   try:
       with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
           for problem in reduced_data:
               f.write(json.dumps(problem, ensure_ascii=False) + '\n')
       print(f"\nReduced dataset saved to: {OUTPUT_FILE}")
   except Exception as e:
       print(f"Error saving file: {e}")
       return
  
   print(f"\n{'='*60}")
   print(f"REDUCTION SUMMARY")
   print(f"{'='*60}")
  
   print(f"Total problems processed: {total_problems}")
   print(f"Problems reduced: {problems_reduced}")
   print(f"Problems unchanged: {total_problems - problems_reduced}")
  
   print(f"\nSolution counts:")
   print(f"Original total solutions: {total_original_solutions}")
   print(f"Reduced total solutions: {total_reduced_solutions}")
   print(f"Solutions removed: {total_original_solutions - total_reduced_solutions}")
   print(f"Reduction ratio: {total_reduced_solutions/total_original_solutions:.1%}")
  
   print(f"\n{'='*60}")
   print(f"ORIGINAL DISTRIBUTION")
   print(f"{'='*60}")
   for count in sorted(original_stats.keys()):
       num_problems = original_stats[count]
       percentage = (num_problems / total_problems) * 100
       print(f"Problems with {count:2d} solutions: {num_problems:3d} ({percentage:5.1f}%)")
  
   print(f"\n{'='*60}")
   print(f"REDUCED DISTRIBUTION")
   print(f"{'='*60}")
   for count in sorted(reduced_stats.keys()):
       num_problems = reduced_stats[count]
       percentage = (num_problems / total_problems) * 100
       print(f"Problems with {count:2d} solutions: {num_problems:3d} ({percentage:5.1f}%)")
  
   print(f"\nAverage solutions per problem:")
   print(f"Original: {total_original_solutions/total_problems:.1f}")
   print(f"Reduced:  {total_reduced_solutions/total_problems:.1f}")

if __name__ == "__main__":
   random.seed(42)
   reduce_balanced_dataset()

