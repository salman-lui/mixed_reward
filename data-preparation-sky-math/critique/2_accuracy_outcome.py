#!/usr/bin/env python3
"""
Calculate verification accuracy for merged verifications by comparing ground truth with outcome verdicts.

This script analyzes how accurately the merged verification system identifies correct vs incorrect solutions.

Accuracy Logic:
- If is_correct="incorrect" AND outcome_verdict="No" → Accurate (correctly identified bad solution)
- If is_correct="correct" AND outcome_verdict="Yes" → Accurate (correctly identified good solution)
- Otherwise → Inaccurate

Note: Field name is automatically extracted from input filename (works for any N value)
"""

import json
from collections import defaultdict
import os

# =============================================================================
# CONFIGURATION - Modify these paths as needed
# =============================================================================
INPUT_FILE = '/code/salsrahm-sandbox/data-reasoning/data-skymath/final_verification_data/without_reference/outcome_level/critique_verdict/verification_final_match_outcome_n_1_with_extracted_verdicts.jsonl'
OUTPUT_FILE = '/code/salsrahm-sandbox/data-reasoning/data-skymath/final_verification_data/without_reference/outcome_level/critique_verdict/verification_final_match_outcome_n_1_merged_verdict_results.json'

# INPUT_FILE = '/code/salsrahm-sandbox/data-reasoning/data-skymath/final_verification_data/without_reference/outcome_level/critique_verdict/verification_final_match_outcome_n_16_with_extracted_verdicts.jsonl'
# OUTPUT_FILE = '/code/salsrahm-sandbox/data-reasoning/data-skymath/final_verification_data/without_reference/outcome_level/critique_verdict/verification_final_match_outcome_n_16_merged_verdict_results.json'


def extract_field_name_from_filename(input_file_path: str) -> str:
   """
   Extract the field name from the input filename.
  
   Example:
   - Input: "verification_final_match_outcome_n_1_merged_verdicts.jsonl"
   - Output: "merged_verification_final_match_outcome_n_1"
   """
   # Get filename without path and extension
   filename = os.path.basename(input_file_path)
   base_name = os.path.splitext(filename)[0]  # Remove .jsonl
  
   # Expected pattern: verification_final_match_outcome_n_X_merged_verdicts
   if "_merged_verdicts" in base_name:
       # Remove "_merged_verdicts" suffix
       core_name = base_name.replace("_merged_verdicts", "")
       # Add "merged_" prefix
       field_name = f"merged_{core_name}"
       return field_name
   else:
       # Fallback to default if pattern doesn't match
       print(f"Warning: Filename pattern not recognized, using default field name")
       return "merged_verification_final_match_outcome_n_1"

def load_data_with_merged_verifications(input_file):
   """Load JSONL data with merged verification outcomes."""
   problems = []
  
   print(f"Loading data from: {input_file}")
   with open(input_file, 'r') as f:
       for line_num, line in enumerate(f, 1):
           try:
               problem = json.loads(line.strip())
               problems.append(problem)
           except json.JSONDecodeError as e:
               print(f"Warning: Failed to parse line {line_num}: {e}")
               continue
  
   print(f"Loaded {len(problems)} problems")
   return problems

def calculate_merged_verification_accuracy(problems, merged_field_name):
   """Calculate verification accuracy for merged verifications."""
  
   # Initialize accuracy tracking
   accuracy_stats = {
       'total_solutions': 0,
       'accurate_verifications': 0,
       'inaccurate_verifications': 0,
       'breakdown': {
           'correct_solutions': {
               'total': 0,
               'correctly_verified_yes': 0,  # is_correct=correct, outcome_verdict=Yes
               'wrongly_verified_no': 0      # is_correct=correct, outcome_verdict=No
           },
           'incorrect_solutions': {
               'total': 0,
               'correctly_verified_no': 0,   # is_correct=incorrect, outcome_verdict=No
               'wrongly_verified_yes': 0     # is_correct=incorrect, outcome_verdict=Yes
           }
       }
   }
  
   # Initialize example indices collection for manual investigation
   example_indices = {
       'correct_verified_yes': [],      # Correctly identified correct solutions
       'incorrect_verified_no': [],     # Correctly identified incorrect solutions 
       'correct_verified_no': [],       # Incorrectly rejected correct solutions
       'incorrect_verified_yes': []     # Incorrectly approved incorrect solutions
   }
  
   print("\n" + "="*80)
   print("CALCULATING MERGED VERIFICATION ACCURACY")
   print("="*80)
  
   total_problems = len(problems)
   processed_solutions = 0
  
   for problem_idx, problem in enumerate(problems):
       # Get ground truth (at problem level)
       is_correct = problem.get('is_correct', '')
      
       # Skip if missing ground truth
       if not is_correct or is_correct not in ['correct', 'incorrect']:
           continue
          
       # Get merged verification outcomes
       merged_verifications = problem.get(merged_field_name, [])
      
       if not merged_verifications:
           continue
      
       # Process first merged verification (should be only one)
       merged_verification = merged_verifications[0] if merged_verifications else {}
       outcome_verdict = merged_verification.get('outcome_verdict', '')
      
       # Skip if missing outcome verdict
       if not outcome_verdict or outcome_verdict not in ['Yes', 'No']:
           continue
      
       processed_solutions += 1
       accuracy_stats['total_solutions'] += 1
      
       # Determine if verification is accurate
       is_accurate = False
      
       if is_correct == 'incorrect' and outcome_verdict == 'No':
           # Correctly identified bad solution
           is_accurate = True
           accuracy_stats['breakdown']['incorrect_solutions']['correctly_verified_no'] += 1
           # Collect example indices (limit to 3)
           if len(example_indices['incorrect_verified_no']) < 3:
               example_indices['incorrect_verified_no'].append(problem_idx)
       elif is_correct == 'correct' and outcome_verdict == 'Yes':
           # Correctly identified good solution 
           is_accurate = True
           accuracy_stats['breakdown']['correct_solutions']['correctly_verified_yes'] += 1
           # Collect example indices (limit to 3)
           if len(example_indices['correct_verified_yes']) < 3:
               example_indices['correct_verified_yes'].append(problem_idx)
       elif is_correct == 'incorrect' and outcome_verdict == 'Yes':
           # Wrongly approved bad solution
           is_accurate = False
           accuracy_stats['breakdown']['incorrect_solutions']['wrongly_verified_yes'] += 1
           # Collect example indices (limit to 3)
           if len(example_indices['incorrect_verified_yes']) < 3:
               example_indices['incorrect_verified_yes'].append(problem_idx)
       elif is_correct == 'correct' and outcome_verdict == 'No':
           # Wrongly rejected good solution
           is_accurate = False
           accuracy_stats['breakdown']['correct_solutions']['wrongly_verified_no'] += 1
           # Collect example indices (limit to 3)
           if len(example_indices['correct_verified_no']) < 3:
               example_indices['correct_verified_no'].append(problem_idx)
      
       # Update accuracy counters
       if is_accurate:
           accuracy_stats['accurate_verifications'] += 1
       else:
           accuracy_stats['inaccurate_verifications'] += 1
      
       # Update solution type counters
       if is_correct == 'correct':
           accuracy_stats['breakdown']['correct_solutions']['total'] += 1
       else:
           accuracy_stats['breakdown']['incorrect_solutions']['total'] += 1
  
   print(f"Total problems processed: {total_problems}")
   print(f"Total solutions analyzed: {processed_solutions}")
  
   return accuracy_stats, example_indices

def print_example_indices(example_indices):
   """Print example indices for manual investigation."""
  
   print("\n" + "="*80)
   print("EXAMPLE INDICES FOR MANUAL INVESTIGATION (MERGED VERIFICATIONS)")
   print("="*80)
  
   print(f"\n✅ Correctly identified CORRECT solutions (is_correct='correct', outcome_verdict='Yes'):")
   indices = example_indices['correct_verified_yes']
   if indices:
       print(f"   Indices: {indices}")
   else:
       print(f"   No examples found")
      
   print(f"\n✅ Correctly identified INCORRECT solutions (is_correct='incorrect', outcome_verdict='No'):")
   indices = example_indices['incorrect_verified_no']
   if indices:
       print(f"   Indices: {indices}")
   else:
       print(f"   No examples found")
      
   print(f"\n❌ Incorrectly rejected CORRECT solutions (is_correct='correct', outcome_verdict='No'):")
   indices = example_indices['correct_verified_no']
   if indices:
       print(f"   Indices: {indices}")
   else:
       print(f"   No examples found")
      
   print(f"\n❌ Incorrectly approved INCORRECT solutions (is_correct='incorrect', outcome_verdict='Yes'):")
   indices = example_indices['incorrect_verified_yes']
   if indices:
       print(f"   Indices: {indices}")
   else:
       print(f"   No examples found")

def print_merged_accuracy_results(accuracy_stats):
   """Print detailed merged verification accuracy results."""
  
   print("\n" + "="*80)
   print("MERGED VERIFICATION ACCURACY RESULTS")
   print("="*80)
  
   total = accuracy_stats.get('total_solutions', 0)
   accurate = accuracy_stats.get('accurate_verifications', 0)
   inaccurate = accuracy_stats.get('inaccurate_verifications', 0)
  
   if total > 0:
       accuracy_pct = (accurate / total) * 100
       print(f"\n{'Metric':<25} {'Count':<10} {'Percentage':<12}")
       print("-" * 50)
       print(f"{'Total Solutions':<25} {total:<10} {'100.0%':<12}")
       print(f"{'Accurate Verifications':<25} {accurate:<10} {accuracy_pct:<12.2f}%")
       print(f"{'Inaccurate Verifications':<25} {inaccurate:<10} {100-accuracy_pct:<12.2f}%")
   else:
       print("No solutions to analyze")
       return
  
   # Breakdown by solution correctness
   correct_sols = accuracy_stats['breakdown']['correct_solutions']
   incorrect_sols = accuracy_stats['breakdown']['incorrect_solutions']
  
   print(f"\n" + "="*60)
   print(f"DETAILED BREAKDOWN")
   print("="*60)
  
   print(f"\nCorrect Solutions (Ground Truth = 'correct'):")
   print(f"  Total: {correct_sols['total']}")
   print(f"  Correctly verified as Yes: {correct_sols['correctly_verified_yes']}")
   print(f"  Wrongly verified as No: {correct_sols['wrongly_verified_no']}")
   if correct_sols['total'] > 0:
       correct_accuracy = (correct_sols['correctly_verified_yes'] / correct_sols['total']) * 100
       print(f"  Accuracy on correct solutions: {correct_accuracy:.2f}%")
  
   print(f"\nIncorrect Solutions (Ground Truth = 'incorrect'):")
   print(f"  Total: {incorrect_sols['total']}")
   print(f"  Correctly verified as No: {incorrect_sols['correctly_verified_no']}")
   print(f"  Wrongly verified as Yes: {incorrect_sols['wrongly_verified_yes']}")
   if incorrect_sols['total'] > 0:
       incorrect_accuracy = (incorrect_sols['correctly_verified_no'] / incorrect_sols['total']) * 100
       print(f"  Accuracy on incorrect solutions: {incorrect_accuracy:.2f}%")

def save_merged_accuracy_results(accuracy_stats, output_file, merged_field_name):
   """Save merged verification accuracy results to JSON file."""
  
   print(f"\nSaving merged verification accuracy results to: {output_file}")
  
   # Convert to serializable format
   results = {
       'merged_verification_accuracy_analysis': accuracy_stats,
       'field_name_used': merged_field_name,
       'summary': {}
   }
  
   # Add summary statistics
   total = accuracy_stats.get('total_solutions', 0)
   accurate = accuracy_stats.get('accurate_verifications', 0)
  
   results['summary'] = {
       'total_solutions': total,
       'accurate_verifications': accurate,
       'accuracy_percentage': (accurate / total * 100) if total > 0 else 0
   }
  
   # Create output directory if it doesn't exist
   os.makedirs(os.path.dirname(output_file), exist_ok=True)
  
   with open(output_file, 'w') as f:
       json.dump(results, f, indent=2)
  
   print(f"Merged verification accuracy results saved successfully!")

def main():
   """Main function to calculate merged verification accuracy."""
  
   # Ensure input file exists
   if not os.path.exists(INPUT_FILE):
       print(f"Error: Input file not found: {INPUT_FILE}")
       return
  
   # Extract field name from input filename
   merged_field_name = extract_field_name_from_filename(INPUT_FILE)
   print(f"Using field name: {merged_field_name}")
  
   # Load data
   problems = load_data_with_merged_verifications(INPUT_FILE)
  
   if not problems:
       print("Error: No data loaded")
       return
  
   # Calculate accuracy
   accuracy_stats, example_indices = calculate_merged_verification_accuracy(problems, merged_field_name)
  
   # Print results
   print_merged_accuracy_results(accuracy_stats)
  
   # Print example indices for manual investigation
   print_example_indices(example_indices)
  
   # Create dynamic output filename
   input_basename = os.path.basename(INPUT_FILE)
   output_basename = input_basename.replace("_verdicts.jsonl", "_accuracy_results.json")
   dynamic_output_file = os.path.join(os.path.dirname(OUTPUT_FILE), output_basename)
  
   # Save results
   save_merged_accuracy_results(accuracy_stats, dynamic_output_file, merged_field_name)
  
   print(f"\n" + "="*80)
   print("MERGED VERIFICATION ACCURACY ANALYSIS COMPLETED!")
   print("="*80)

if __name__ == "__main__":
   main()

