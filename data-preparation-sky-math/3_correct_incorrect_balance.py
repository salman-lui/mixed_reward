INPUT_FILE = '/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/skywork_solutions_generated_eval_with_verification.jsonl'
OUTPUT_FILE = '/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/skywork_solutions_generated_eval_with_verification_balanced.jsonl'

import json
import random
from collections import defaultdict

def balance_solutions_data():
   """Balance dataset by keeping equal numbers of correct and incorrect solutions per problem"""
   with open(INPUT_FILE, 'r') as f:
       data = [json.loads(line) for line in f]
  
   balanced_data = []
   total_problems = len(data)
   kept_problems = 0
   total_original_solutions = 0
   total_balanced_solutions = 0
  
   print(f"Processing {total_problems} problems...")
  
   for problem_idx, item in enumerate(data, 1):
       solutions = item.get('generated_solutions', [])
       total_original_solutions += len(solutions)
      
       correct_solutions = []
       incorrect_solutions = []
      
       for solution in solutions:
           if solution.get('is_correct') == 'correct':
               correct_solutions.append(solution)
           else:
               incorrect_solutions.append(solution)
      
       correct_count = len(correct_solutions)
       incorrect_count = len(incorrect_solutions)
      
       print(f"Problem {problem_idx}: {correct_count} correct, {incorrect_count} incorrect")
      
       if correct_count == 0 or incorrect_count == 0:
           print(f"  -> Skipping (all solutions have same correctness)")
           continue
      
       balance_count = min(correct_count, incorrect_count)
      
       random.shuffle(correct_solutions)
       random.shuffle(incorrect_solutions)
      
       selected_correct = correct_solutions[:balance_count]
       selected_incorrect = incorrect_solutions[:balance_count]
      
       balanced_solutions = selected_correct + selected_incorrect
       total_balanced_solutions += len(balanced_solutions)
      
       balanced_item = item.copy()
       balanced_item['generated_solutions'] = balanced_solutions
      
       balanced_data.append(balanced_item)
       kept_problems += 1
      
       print(f"  -> Kept: {balance_count} correct + {balance_count} incorrect = {len(balanced_solutions)} total")
  
   with open(OUTPUT_FILE, 'w') as f:
       for item in balanced_data:
           f.write(json.dumps(item) + '\n')
  
   print(f"\n{'='*50}")
   print(f"BALANCING SUMMARY")
   print(f"{'='*50}")
   print(f"Original problems: {total_problems}")
   print(f"Kept problems: {kept_problems}")
   print(f"Excluded problems: {total_problems - kept_problems}")
   print(f"Original solutions: {total_original_solutions}")
   print(f"Balanced solutions: {total_balanced_solutions}")
   print(f"Reduction ratio: {total_balanced_solutions}/{total_original_solutions} = {total_balanced_solutions/total_original_solutions:.2%}")
   print(f"\nBalanced data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
   random.seed(42)
   balance_solutions_data()

