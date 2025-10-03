#!/usr/bin/env python3
"""
Step & Outcome Verdict Extraction for Merged Verifications

This script processes merged verification data to extract:
1. Step-level verdicts (correct/incorrect for each step)
2. Outcome-level verdicts (Yes/No for final answer correctness)

Input: JSONL with merged_verification_final_match_outcome_n_X containing merged verification text
Output: Enhanced JSONL with step_verdicts and outcome_verdict fields for merged verifications
Note: Field name is automatically extracted from input filename (works for any N value)
"""

import json
import re
from typing import Dict, Optional, Any
import os

# Configuration
INPUT_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/final_verification_data/without_reference/outcome_level/verification_final_match_outcome_n_1_merged.jsonl"
OUTPUT_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/final_verification_data/without_reference/outcome_level/critique_verdict/verification_final_match_outcome_n_1_with_extracted_verdicts.jsonl"

def extract_field_name_from_filename(input_file_path: str) -> str:
   """
   Extract the field name from the input filename.
  
   Example:
   - Input: "verification_final_match_outcome_n_1_merged.jsonl"
   - Output: "merged_verification_final_match_outcome_n_1"
   """
   # Get filename without path and extension
   filename = os.path.basename(input_file_path)
   base_name = os.path.splitext(filename)[0]  # Remove .jsonl
  
   # Expected pattern: verification_final_match_outcome_n_X_merged
   if base_name.endswith("_merged"):
       # Remove "_merged" suffix
       core_name = base_name[:-7]  # Remove "_merged"
       # Add "merged_" prefix
       field_name = f"merged_{core_name}"
       return field_name
   else:
       # Fallback to default if pattern doesn't match
       print(f"Warning: Filename pattern not recognized, using default field name")
       return "merged_verification_final_match_outcome_n_1"

def load_all_problems(file_path: str) -> list:
   """Load all problems from JSONL file."""
   problems = []
   try:
       with open(file_path, 'r', encoding='utf-8') as f:
           for line_num, line in enumerate(f):
               try:
                   problem_data = json.loads(line.strip())
                   problems.append(problem_data)
               except json.JSONDecodeError as e:
                   print(f"Error parsing line {line_num + 1}: {e}")
                   continue
      
       print(f"✅ Loaded {len(problems)} problems successfully")
       return problems
   except FileNotFoundError:
       print(f"Error: File {file_path} not found")
       return []
   except Exception as e:
       print(f"Error loading problems: {e}")
       return []

def extract_step_verdicts(verification_text: str) -> Dict[str, str]:
   """
   Extract step-by-step verdicts from verification text.
   Returns: dict {"Step 1": "correct", "Step 2": "incorrect", ...}
   """
   step_verdicts = {}
  
   # Remove <think> sections if present
   if "</think>" in verification_text:
       verification_text = verification_text.split("</think>", 1)[1]
   else:
       # Special case: LLM forgot to close </think> tag
       # Look for where ## Step 1: actually starts
       step_start_pattern = r'\n\n## Step 1:'
       if re.search(step_start_pattern, verification_text):
           # Split at this pattern and take everything from ## Step 1:
           parts = re.split(r'\n\n(?=## Step 1:)', verification_text, 1)
           if len(parts) > 1:
               verification_text = parts[1]
  
   # Find all step patterns: ## Step X: ... **This step is correct/incorrect.**
   step_pattern = r'## Step (\d+):.*?\*\*This step is (correct|incorrect)\.\*\*'
   matches = re.findall(step_pattern, verification_text, re.DOTALL | re.IGNORECASE)
  
   # If no matches found with ## pattern, try **Step pattern as fallback
   if not matches:
       step_pattern_fallback = r'\*\*Step (\d+):.*?\*\*This step is (correct|incorrect)\.\*\*'
       matches = re.findall(step_pattern_fallback, verification_text, re.DOTALL | re.IGNORECASE)
  
   for step_num, verdict in matches:
       step_verdicts[f"Step {step_num}"] = verdict.lower()
  
   return step_verdicts

def extract_outcome_verdict(verification_text: str) -> str:
   """
   Extract Yes/No verdict from verification text.
   Returns: "Yes", "No", or "failed" if not found
   """
   # Remove <think> sections if present
   if "</think>" in verification_text:
       verification_text = verification_text.split("</think>", 1)[1]
   else:
       # Special case: LLM forgot to close </think> tag
       # Look for where the verification actually starts (after thinking)
       verification_start_patterns = [
           (r'\n\n## Step 1:', r'\n\n(?=## Step 1:)'),
           (r'\n\n\*\*Verification:', r'\n\n(?=\*\*Verification:)'),
           (r'\n\nVerification:', r'\n\n(?=Verification:)'),
           (r'\n\nIs the answer correct', r'\n\n(?=Is the answer correct)')
       ]
      
       for search_pattern, split_pattern in verification_start_patterns:
           if re.search(search_pattern, verification_text):
               # Split at this pattern and take everything from the verification start
               parts = re.split(split_pattern, verification_text, 1)
               if len(parts) > 1:
                   verification_text = parts[1]
                   break
  
   # Multiple patterns to catch different formats
   patterns = [
       r"Verification:\s*Is the answer correct.*?\?\s*(Yes|No)",
       r"Is the answer correct.*?\?\s*(Yes|No)",
       r"answer correct.*?\?\s*(Yes|No)",
       r"\*\*Verification:\s*Is the answer correct.*?\?\s*(Yes|No)\*\*",
       r"\?\s*(Yes|No)\s*\*\*\s*$",
       r"\?\s*(Yes|No)\s*$"
   ]
  
   for pattern in patterns:
       match = re.search(pattern, verification_text, re.IGNORECASE)
       if match:
           verdict = match.group(1)
           return verdict  # Return "Yes" or "No" as found
  
   return "failed"  # Could not extract verdict

def process_merged_verifications():
   """Process all problems and extract verdicts from merged verifications."""
  
   print(f"Loading data from: {INPUT_FILE}")
   problems = load_all_problems(INPUT_FILE)
  
   if not problems:
       print("No problems loaded. Exiting.")
       return
  
   # Extract field name from input filename
   merged_field_name = extract_field_name_from_filename(INPUT_FILE)
   print(f"Using field name: {merged_field_name}")
  
   # Create dynamic output filename based on input
   input_basename = os.path.basename(INPUT_FILE)
   output_basename = input_basename.replace("_merged.jsonl", "_with_extracted_verdicts.jsonl")
   dynamic_output_file = os.path.join(os.path.dirname(OUTPUT_FILE), output_basename)
   print(f"Output will be saved to: {dynamic_output_file}")
  
   # Create output directory if it doesn't exist
   os.makedirs(os.path.dirname(dynamic_output_file), exist_ok=True)
  
   total_merged_verifications = 0
   successful_step_extractions = 0
   successful_outcome_extractions = 0
  
   # Process each problem
   for problem_idx, problem in enumerate(problems, 1):
       print(f"Processing problem {problem_idx}...")
      
       # Get merged verification using dynamic field name
       merged_verification_list = problem.get(merged_field_name, [])
      
       if not merged_verification_list:
           print(f"  No merged verification found in field '{merged_field_name}' - skipping")
           continue
      
       # Process each merged verification (usually just one)
       for merged_verification in merged_verification_list:
           merged_verification_text = merged_verification.get("merged_verification_text", "")
          
           if not merged_verification_text:
               merged_verification["step_verdicts"] = {}
               merged_verification["outcome_verdict"] = "failed"
               continue
          
           # Extract step verdicts
           step_verdicts = extract_step_verdicts(merged_verification_text)
           merged_verification["step_verdicts"] = step_verdicts
          
           # Extract outcome verdict
           outcome_verdict = extract_outcome_verdict(merged_verification_text)
           merged_verification["outcome_verdict"] = outcome_verdict
          
           # Update statistics
           total_merged_verifications += 1
           if step_verdicts:
               successful_step_extractions += 1
           if outcome_verdict != "failed":
               successful_outcome_extractions += 1
          
           # Debug info for first few verifications
           if total_merged_verifications <= 3:
               print(f"  Merged Verification {merged_verification.get('merged_verification_id', '?')}: "
                     f"Steps={len(step_verdicts)}, Outcome={outcome_verdict}")
               print(f"  Step verdicts: {step_verdicts}")
  
   # Save enhanced dataset
   try:
       with open(dynamic_output_file, 'w', encoding='utf-8') as f:
           for problem in problems:
               f.write(json.dumps(problem, ensure_ascii=False) + '\n')
       print(f"Enhanced dataset saved to: {dynamic_output_file}")
   except Exception as e:
       print(f"Error saving enhanced dataset: {e}")
       return
  
   # Print statistics
   print(f"\n{'='*60}")
   print(f"MERGED VERIFICATION EXTRACTION SUMMARY")
   print(f"{'='*60}")
   print(f"Total problems processed: {len(problems)}")
   print(f"Total merged verifications processed: {total_merged_verifications}")
   print(f"Successful step extractions: {successful_step_extractions} ({successful_step_extractions/total_merged_verifications*100:.1f}%)" if total_merged_verifications > 0 else "Successful step extractions: 0")
   print(f"Successful outcome extractions: {successful_outcome_extractions} ({successful_outcome_extractions/total_merged_verifications*100:.1f}%)" if total_merged_verifications > 0 else "Successful outcome extractions: 0")
  
   # Show sample of outcomes
   outcome_counts = {"Yes": 0, "No": 0, "failed": 0}
   for problem in problems:
       for merged_verification in problem.get(merged_field_name, []):
           outcome = merged_verification.get("outcome_verdict", "failed")
           outcome_counts[outcome] += 1
  
   print(f"\nMerged verification outcome verdict distribution:")
   for outcome, count in outcome_counts.items():
       percentage = (count / total_merged_verifications * 100) if total_merged_verifications > 0 else 0
       print(f"  {outcome}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
   process_merged_verifications()

