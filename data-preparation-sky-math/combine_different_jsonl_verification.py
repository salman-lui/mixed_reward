#!/usr/bin/env python3
"""Combine multiple JSONL verification files into a single output file"""

import json
import os
from collections import defaultdict

INPUT_FILES = [
   "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/with_reference_verification/part-1-done-verification/skywork_verifications_with_reference_train_part_1.jsonl",
   "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/with_reference_verification/part-1-done-verification/skywork_verifications_with_reference_train_part_3.jsonl",
   "/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/reduced_balanced_data/with_reference_verification/part-1-done-verification/skywork_verifications_with_reference_train_part_4.jsonl"
]

OUTPUT_FILE = "/code/salsrahm-sandbox/data-reasoning/data-skymath/final_verification_data/with_reference/final_skywork_verifications_with_reference_train.jsonl"

def validate_input_files(input_files):
   """Check which input files exist and return list of existing files"""
   print("VALIDATING INPUT FILES:")
   print("=" * 60)
  
   existing_files = []
   missing_files = []
  
   for i, file_path in enumerate(input_files, 1):
       if os.path.exists(file_path):
           file_size = os.path.getsize(file_path)
           print(f"Input {i}: {file_path}")
           print(f"   Size: {file_size:,} bytes")
           existing_files.append(file_path)
       else:
           print(f"Input {i}: {file_path}")
           print(f"   Status: FILE NOT FOUND")
           missing_files.append(file_path)
  
   print(f"\nVALIDATION SUMMARY:")
   print(f"  Existing files: {len(existing_files)}")
   print(f"  Missing files:  {len(missing_files)}")
  
   if missing_files:
       print(f"\nWARNING: {len(missing_files)} files are missing!")
       for file_path in missing_files:
           print(f"    - {file_path}")
       print(f"\nContinuing with {len(existing_files)} existing files...")
  
   return existing_files

def analyze_file_content(file_path):
   """Analyze a single JSONL file and return statistics"""
   stats = {
       "file_path": file_path,
       "total_records": 0,
       "verification_counts": defaultdict(int),
       "data_sources": defaultdict(int),
       "parse_errors": 0
   }
  
   try:
       with open(file_path, 'r', encoding='utf-8') as f:
           for line_num, line in enumerate(f, 1):
               try:
                   record = json.loads(line.strip())
                   stats["total_records"] += 1
                  
                   # Count verifications
                   num_verifications = len(record.get('verifications', []))
                   stats["verification_counts"][num_verifications] += 1
                  
                   # Count data sources
                   data_source = record.get('data_source', 'unknown')
                   stats["data_sources"][data_source] += 1
                  
               except json.JSONDecodeError as e:
                   stats["parse_errors"] += 1
                   print(f"Parse error in {file_path} line {line_num}: {e}")
               except Exception as e:
                   stats["parse_errors"] += 1
                   print(f"Error in {file_path} line {line_num}: {e}")
                  
   except Exception as e:
       print(f"Error reading {file_path}: {e}")
       return None
  
   return stats

def combine_jsonl_files(input_files, output_file):
   """Combine multiple JSONL files into a single output file"""
   print(f"\nCOMBINING FILES:")
   print("=" * 60)
  
   file_stats = []
   total_input_records = 0
  
   for i, file_path in enumerate(input_files, 1):
       print(f"\nAnalyzing Input {i}: {os.path.basename(file_path)}")
       stats = analyze_file_content(file_path)
       if stats:
           file_stats.append(stats)
           total_input_records += stats["total_records"]
           print(f"  Records: {stats['total_records']:,}")
           print(f"  Parse errors: {stats['parse_errors']}")
          
           if stats["verification_counts"]:
               print(f"  Verification counts:")
               for count in sorted(stats["verification_counts"].keys())[:5]:
                   print(f"    {count} verifications: {stats['verification_counts'][count]:,} records")
       else:
           print(f"  Failed to analyze file")
  
   print(f"\nCOMBINATION SUMMARY:")
   print(f"  Total input files: {len(input_files)}")
   print(f"  Successfully analyzed: {len(file_stats)}")
   print(f"  Total records to combine: {total_input_records:,}")
  
   output_dir = os.path.dirname(output_file)
   if output_dir:
       os.makedirs(output_dir, exist_ok=True)
  
   print(f"\nWriting combined file to: {output_file}")
   combined_records = 0
  
   try:
       with open(output_file, 'w', encoding='utf-8') as output_f:
           for i, file_path in enumerate(input_files, 1):
               print(f"  Processing Input {i}: {os.path.basename(file_path)}")
              
               if not os.path.exists(file_path):
                   print(f"    Skipping - file not found")
                   continue
              
               file_records = 0
               try:
                   with open(file_path, 'r', encoding='utf-8') as input_f:
                       for line_num, line in enumerate(input_f, 1):
                           try:
                               record = json.loads(line.strip())
                               output_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                               file_records += 1
                               combined_records += 1
                              
                               if file_records % 1000 == 0:
                                   print(f"    Progress: {file_records:,} records")
                                  
                           except json.JSONDecodeError as e:
                               print(f"    Skipping invalid JSON at line {line_num}: {e}")
                           except Exception as e:
                               print(f"    Error at line {line_num}: {e}")
              
                   print(f"    Added {file_records:,} records from {os.path.basename(file_path)}")
                  
               except Exception as e:
                   print(f"    Error reading {file_path}: {e}")
                   continue
      
       print(f"\nCOMBINATION COMPLETE!")
       print(f"=" * 60)
       print(f"FINAL STATISTICS:")
       print(f"  Total records combined: {combined_records:,}")
       print(f"  Output file: {output_file}")
       print(f"  Output file size: {os.path.getsize(output_file):,} bytes")
      
       if combined_records == total_input_records:
           print(f"  SUCCESS: All {total_input_records:,} records combined successfully!")
       else:
           print(f"  WARNING: Expected {total_input_records:,} records, but combined {combined_records:,}")
      
       return combined_records
      
   except Exception as e:
       print(f"Error writing output file: {e}")
       return 0

def display_configuration():
   """Display current configuration settings"""
   print("JSONL COMBINATION SCRIPT")
   print("=" * 80)
   print(f"Configuration:")
   print(f"  Input files: {len(INPUT_FILES)}")
   for i, file_path in enumerate(INPUT_FILES, 1):
       print(f"    {i}. {file_path}")
   print(f"  Output file: {OUTPUT_FILE}")
   print("=" * 80)

def main():
   """Main function to combine JSONL verification files"""
   display_configuration()
  
   existing_files = validate_input_files(INPUT_FILES)
  
   if not existing_files:
       print("ERROR: No input files found! Please check file paths.")
       return
  
   if len(existing_files) < len(INPUT_FILES):
       response = input(f"\nProceed with {len(existing_files)} files? (y/n): ")
       if response.lower() != 'y':
           print("Operation cancelled.")
           return
  
   combined_count = combine_jsonl_files(existing_files, OUTPUT_FILE)
  
   if combined_count > 0:
       print(f"\nSuccessfully combined {combined_count:,} records!")
       print(f"Combined file ready: {OUTPUT_FILE}")
   else:
       print(f"\nCombination failed!")

if __name__ == "__main__":
   main()

