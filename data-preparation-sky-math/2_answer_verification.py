INPUT_FILE = '/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/skywork_solutions_generated_eval.jsonl'
OUTPUT_FILE = '/code/salsrahm-sandbox/data-reasoning/data-skymath/prepared_data/skywork_solutions_generated_eval_with_verification.jsonl'

import json
import re
from math_verify import parse, verify

def strip_string(string):
   """Normalize mathematical expressions for comparison"""
   if string is None:
       return ""
  
   string = string.replace("\n", "")
   string = string.replace("\\\\", "\\")
   string = string.replace("\\left", "")
   string = string.replace("\\right", "")
   string = string.replace(" ", "")
  
   if len(string.split("=")) == 2:
       if len(string.split("=")[0]) <= 3:
           string = string.split("=")[1]
  
   if "/" in string and len(string.split("/")) == 2:
       parts = string.split("/")
       try:
           int(parts[0])
           int(parts[1])
           string = f"\\frac{{{parts[0]}}}{{{parts[1]}}}"
       except:
           pass
  
   return string.strip()

def is_equiv(str1, str2):
   """Check if two mathematical expressions are equivalent"""
   if str1 is None and str2 is None:
       return True
   if str1 is None or str2 is None:
       return False
  
   try:
       ss1 = strip_string(str1)
       ss2 = strip_string(str2)
       return ss1 == ss2
   except:
       return str1 == str2

def extract_answer_from_solution(solution_str):
   """Extract the final answer from various formats"""
   answer_pattern = r'Answer:\s*(.+?)(?:\n|$)'
   match = re.search(answer_pattern, solution_str, re.IGNORECASE)
   if match:
       return match.group(1).strip()
  
   boxed_pattern = r'\\boxed\{([^}]+)\}'
   matches = re.findall(boxed_pattern, solution_str)
   if matches:
       return matches[-1]
  
   lines = solution_str.strip().split('\n')
   if lines:
       last_line = lines[-1].strip()
       if len(last_line) < 50 and any(c in last_line for c in ['=', '$', '\\', '{', '}']):
           return last_line
  
   return None

def verify_solution(solution_str, ground_truth):
   """Verify if a solution is correct using hybrid approach"""
   try:
       if isinstance(ground_truth, str):
           ground_truth = [ground_truth]
      
       cleaned_gt = []
       for gt in ground_truth:
           if gt.startswith('["') and gt.endswith('"]'):
               gt = gt[2:-2]
           cleaned_gt.append(gt)
      
       if "</think>" in solution_str:
           solution_str = solution_str.split("</think>")[1]
      
       extracted_answer = extract_answer_from_solution(solution_str)
      
       if extracted_answer is None:
           return False, "No answer found"
      
       for gt in cleaned_gt:
           if is_equiv(extracted_answer, gt):
               return True, extracted_answer
      
       try:
           parsed_solution = parse(solution_str, parsing_timeout=20)
           if len(parsed_solution) >= 2:
               math_verify_answer = parsed_solution[1]
               for gt in cleaned_gt:
                   try:
                       if verify(
                           parse(f"\\boxed{{{gt}}}", parsing_timeout=20),
                           parsed_solution,
                           timeout_seconds=20,
                       ):
                           return True, extracted_answer
                   except:
                       continue
       except:
           pass
      
       return False, extracted_answer
      
   except Exception as e:
       return False, f"Error: {str(e)}"

def main():
   with open(INPUT_FILE, 'r') as f:
       data = [json.loads(line) for line in f]
  
   for problem_idx, item in enumerate(data, 1):
       print(f"Processing problem {problem_idx}...")
      
       ground_truth = item.get('reward_model', {}).get('ground_truth', [])
       solutions = item.get('generated_solutions', [])
      
       for solution_data in solutions:
           solution_text = solution_data.get('generated_solution', '')
           is_correct, extracted_answer = verify_solution(solution_text, ground_truth)
           solution_data['is_correct'] = "correct" if is_correct else "incorrect"
           solution_data['extracted_answer'] = extracted_answer
  
   with open(OUTPUT_FILE, 'w') as f:
       for item in data:
           f.write(json.dumps(item) + '\n')
  
   print(f"Enhanced data saved to: {OUTPUT_FILE}")
   print(f"Added 'is_correct' and 'extracted_answer' fields to each solution")

if __name__ == "__main__":
   main()

