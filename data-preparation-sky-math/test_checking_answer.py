#!/usr/bin/env python3
"""Analyze solution count distribution in balanced dataset"""

import json
from collections import defaultdict

INPUT_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/skywork_solutions_generated_train_with_verification_balanced.jsonl"

def analyze_solution_distribution():
   """Analyze distribution of solution counts in balanced dataset"""
  
   print(f"Analyzing solution distribution from: {INPUT_FILE}")
  
   solution_counts = defaultdict(int)
   total_problems = 0
   total_solutions = 0
  
   try:
       with open(INPUT_FILE, 'r', encoding='utf-8') as f:
           for line_num, line in enumerate(f, 1):
               try:
                   problem_data = json.loads(line.strip())
                  
                   generated_solutions = problem_data.get("generated_solutions", [])
                   num_solutions = len(generated_solutions)
                  
                   solution_counts[num_solutions] += 1
                   total_problems += 1
                   total_solutions += num_solutions
                  
               except json.JSONDecodeError as e:
                   print(f"Error parsing line {line_num}: {e}")
                   continue
               except Exception as e:
                   print(f"Error processing line {line_num}: {e}")
                   continue
  
   except Exception as e:
       print(f"Error reading file: {e}")
       return
  
   print(f"\n{'='*50}")
   print(f"SOLUTION COUNT DISTRIBUTION")
   print(f"{'='*50}")
  
   target_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 26, 28, 30]
  
   for count in target_counts:
       num_problems = solution_counts[count]
       if num_problems > 0:
           percentage = (num_problems / total_problems) * 100
           print(f"Problems with {count:2d} solutions: {num_problems:3d} ({percentage:5.1f}%)")
       else:
           print(f"Problems with {count:2d} solutions: {num_problems:3d}")
  
   other_counts = set(solution_counts.keys()) - set(target_counts)
   if other_counts:
       print(f"\nOther solution counts found:")
       for count in sorted(other_counts):
           num_problems = solution_counts[count]
           percentage = (num_problems / total_problems) * 100
           print(f"Problems with {count:2d} solutions: {num_problems:3d} ({percentage:5.1f}%)")
  
   print(f"\n{'='*50}")
   print(f"SUMMARY")
   print(f"{'='*50}")
   print(f"Total problems: {total_problems}")
   print(f"Total solutions: {total_solutions}")
   print(f"Average solutions per problem: {total_solutions/total_problems:.1f}")

if __name__ == "__main__":
   analyze_solution_distribution()

