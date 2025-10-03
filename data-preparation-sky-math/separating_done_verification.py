#!/usr/bin/env python3
"""Separate completed verifications from incomplete solutions for continued processing"""

import json
import os
from collections import defaultdict

TARGET_NUM_VERIFICATIONS = 16

INPUT_DIR = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/with_reference_verification/remaining-data-verification/skywork_solutions_remaining_train.jsonl"
OUTPUT_DIR = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/with_reference_verification/part-1-done-verification/skywork_verifications_with_reference_train_part_2.jsonl"
REMAINING_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/with_reference_verification/remaining-data-verification/skywork_solutions_remaining_train2.jsonl"
CLEANED_OUTPUT_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/with_reference_verification/part-1-done-verification/skywork_verifications_with_reference_train_part_3.jsonl"

SOLUTION_FIELD = "solution"
TIMESTAMP_FIELD = "generated_at"

def load_completed_solutions(output_file):
   """Load completed solutions and return set of unique keys (solution, generated_at)"""
   print(f"Loading completed verifications from: {output_file}")
  
   completed_keys = set()
   total_records = 0
   completed_count = 0
   verification_stats = defaultdict(int)
  
   try:
       with open(output_file, 'r', encoding='utf-8') as f:
           for line_num, line in enumerate(f, 1):
               try:
                   record = json.loads(line.strip())
                   total_records += 1
                  
                   num_verifications = record.get('num_verifications_generated', 0)
                   verification_stats[num_verifications] += 1
                  
                   if num_verifications >= TARGET_NUM_VERIFICATIONS:
                       solution_text = record.get(SOLUTION_FIELD, "")
                       timestamp = record.get(TIMESTAMP_FIELD, "")
                      
                       if solution_text and timestamp:
                           key = (solution_text, timestamp)
                           completed_keys.add(key)
                           completed_count += 1
                       else:
                           print(f"Warning: Missing solution or timestamp at line {line_num}")
                  
               except json.JSONDecodeError as e:
                   print(f"Error parsing line {line_num}: {e}")
                   continue
               except Exception as e:
                   print(f"Error processing line {line_num}: {e}")
                   continue
      
       print(f"Loaded {total_records:,} verification records")
       print(f"Found {completed_count:,} completed solutions (>= {TARGET_NUM_VERIFICATIONS} verifications)")
      
       print(f"\nVerification Count Distribution:")
       for count in sorted(verification_stats.keys()):
           print(f"   {count:2d} verifications: {verification_stats[count]:,} solutions")
      
       return completed_keys
      
   except FileNotFoundError:
       print(f"Error: Output file not found: {output_file}")
       return set()
   except Exception as e:
       print(f"Error reading output file: {e}")
       return set()

def filter_remaining_solutions(input_file, completed_keys, output_file):
   """Filter input solutions for incomplete verifications and save remaining solutions"""
   print(f"\nProcessing input solutions from: {input_file}")
  
   remaining_solutions = []
   total_input = 0
   matched_completed = 0
   missing_fields = 0
  
   try:
       with open(input_file, 'r', encoding='utf-8') as f:
           for line_num, line in enumerate(f, 1):
               try:
                   solution = json.loads(line.strip())
                   total_input += 1
                  
                   # Create key for this solution
                   solution_text = solution.get(SOLUTION_FIELD, "")
                   timestamp = solution.get(TIMESTAMP_FIELD, "")
                  
                   if not solution_text or not timestamp:
                       missing_fields += 1
                       print(f"Warning: Missing solution or timestamp at line {line_num}")
                       remaining_solutions.append(solution)
                       continue
                  
                   key = (solution_text, timestamp)
                  
                   if key in completed_keys:
                       matched_completed += 1
                   else:
                       remaining_solutions.append(solution)
                  
               except json.JSONDecodeError as e:
                   print(f"Error parsing line {line_num}: {e}")
                   continue
               except Exception as e:
                   print(f"Error processing line {line_num}: {e}")
                   continue
      
       print(f"\nSaving remaining solutions to: {output_file}")
       output_dir = os.path.dirname(output_file)
       if output_dir:
           os.makedirs(output_dir, exist_ok=True)
      
       with open(output_file, 'w', encoding='utf-8') as f:
           for solution in remaining_solutions:
               f.write(json.dumps(solution, ensure_ascii=False) + '\n')
      
       print(f"\nPROCESSING SUMMARY:")
       print(f"=" * 60)
       print(f"Total input solutions:           {total_input:,}")
       print(f"Matched completed solutions:     {matched_completed:,}")
       print(f"Remaining solutions to process:  {len(remaining_solutions):,}")
       print(f"Solutions with missing fields:   {missing_fields:,}")
       print(f"=" * 60)
      
       expected_remaining = total_input - matched_completed
       if len(remaining_solutions) == expected_remaining:
           print(f"VALIDATION PASSED: Numbers add up correctly!")
       else:
           print(f"VALIDATION WARNING: Expected {expected_remaining:,} remaining, got {len(remaining_solutions):,}")
      
       return len(remaining_solutions)
      
   except FileNotFoundError:
       print(f"Error: Input file not found: {input_file}")
       return 0
   except Exception as e:
       print(f"Error processing input file: {e}")
       return 0

def save_cleaned_output(output_file, cleaned_file):
   """Save cleaned version with only complete verifications"""
   print(f"\nCreating cleaned output file with only complete verifications...")
   print(f"Reading from: {output_file}")
   print(f"Saving cleaned to: {cleaned_file}")
  
   complete_solutions = []
   total_output_records = 0
   incomplete_discarded = 0
  
   try:
       with open(output_file, 'r', encoding='utf-8') as f:
           for line_num, line in enumerate(f, 1):
               try:
                   record = json.loads(line.strip())
                   total_output_records += 1
                  
                   num_verifications = record.get('num_verifications_generated', 0)
                  
                   if num_verifications >= TARGET_NUM_VERIFICATIONS:
                       complete_solutions.append(record)
                   else:
                       incomplete_discarded += 1
                  
               except json.JSONDecodeError as e:
                   print(f"Error parsing line {line_num}: {e}")
                   continue
               except Exception as e:
                   print(f"Error processing line {line_num}: {e}")
                   continue
      
       cleaned_dir = os.path.dirname(cleaned_file)
       if cleaned_dir:
           os.makedirs(cleaned_dir, exist_ok=True)
      
       with open(cleaned_file, 'w', encoding='utf-8') as f:
           for record in complete_solutions:
               f.write(json.dumps(record, ensure_ascii=False) + '\n')
      
       print(f"\nCLEANED OUTPUT SUMMARY:")
       print(f"=" * 60)
       print(f"Total output records processed:     {total_output_records:,}")
       print(f"Complete verifications kept:        {len(complete_solutions):,}")
       print(f"Incomplete verifications discarded: {incomplete_discarded:,}")
       completion_rate = (len(complete_solutions)/total_output_records)*100 if total_output_records > 0 else 0.0
       print(f"Completion rate:                    {completion_rate:.1f}%")
       print(f"=" * 60)
       print(f"Cleaned output saved to: {cleaned_file}")
      
       return len(complete_solutions)
      
   except FileNotFoundError:
       print(f"Error: Output file not found: {output_file}")
       return 0
   except Exception as e:
       print(f"Error processing output file: {e}")
       return 0

def analyze_matching_quality(input_file, completed_keys):
   """Analyze matching quality to detect potential issues"""
   print(f"\nMATCHING QUALITY ANALYSIS:")
   print(f"=" * 60)
  
   print(f"Completed solutions sample (first 3):")
   for i, key in enumerate(list(completed_keys)[:3]):
       solution_preview = key[0][:100] + "..." if len(key[0]) > 100 else key[0]
       print(f"  {i+1}. Solution: {solution_preview}")
       print(f"     Timestamp: {key[1]}")
  
   print(f"\nTotal unique completed keys: {len(completed_keys):,}")

def main():
   """Main function to separate completed and remaining solutions"""
   print("VERIFICATION SEPARATION SCRIPT")
   print("=" * 80)
   print(f"Configuration:")
   print(f"  Target verifications: {TARGET_NUM_VERIFICATIONS}")
   print(f"  Input file: {INPUT_DIR}")
   print(f"  Output file: {OUTPUT_DIR}")
   print(f"  Remaining file: {REMAINING_FILE}")
   print(f"  Cleaned output file: {CLEANED_OUTPUT_FILE}")
   print(f"  Matching fields: ({SOLUTION_FIELD}, {TIMESTAMP_FIELD})")
   print("=" * 80)
  
   completed_keys = load_completed_solutions(OUTPUT_DIR)
  
   if not completed_keys:
       print("No completed solutions found. Nothing to separate.")
       return
  
   analyze_matching_quality(INPUT_DIR, completed_keys)
  
   remaining_count = filter_remaining_solutions(INPUT_DIR, completed_keys, REMAINING_FILE)
  
   complete_count = save_cleaned_output(OUTPUT_DIR, CLEANED_OUTPUT_FILE)
  
   print(f"\nSEPARATION COMPLETE!")
   print(f"=" * 80)
   print(f"FILES CREATED:")
   print(f"  - Remaining input solutions:  {REMAINING_FILE}")
   print(f"    ({remaining_count:,} solutions need processing)")
   print(f"  - Complete verifications:     {CLEANED_OUTPUT_FILE}")
   print(f"    ({complete_count:,} solutions with {TARGET_NUM_VERIFICATIONS}+ verifications)")
   print(f"=" * 80)
   print(f"Ready to continue processing {remaining_count:,} remaining solutions!")
   print(f"You have {complete_count:,} solutions with complete verifications!")

if __name__ == "__main__":
   main()

