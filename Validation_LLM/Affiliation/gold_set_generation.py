import os
import csv
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

curr_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(curr_dir, "affiliations_raw.csv")
output_file = os.path.join(curr_dir, "gold_set.csv")

# System prompt for normalization
SYSTEM_PROMPT = """You are an expert in data normalization and entity resolution. Your task is to normalize raw affiliation strings to standardized organization names.
You will be provided with a mapping dictionary of raw affiliation variants and their corresponding normalized names.
Your goal is to match a given input affiliation string to the correct normalized affiliation,
using the rules implied by the mapping.

Rules to follow:
1. Normalize abbreviations and acronyms to full institution names.
2. Handle punctuation, spacing, and capitalization inconsistencies.
3. Recognize when brand names or departments belong to a parent company or university and map accordingly.
4. Account for multilingual or localized versions of university or organization names.
5. If an affiliation cannot be matched to any known mapping, return "Unknown".

Input:
You will receive a string representing a person's affiliation.

Output:
Return the normalized name. Do not add any additional text in the output please.

Example:
If the input is "UC Berkeley", return "University of California, Berkeley".
If the input is "ATT", return "AT&T".
If the input is "Futurewei", return "Huawei".
If the input is "Independent researcher", return "Unknown".

Use fuzzy matching or logical rules if needed, but ensure high precision."""


def normalize_affiliation(affiliation, model="gpt-4.1", temperature=0):
    """Send affiliation to GPT for normalization"""
    try:
        user_prompt = f"""Task:
Normalize these affiliations:
{affiliation}"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        # Remove markdown formatting if present
        if result.startswith('```json'):
            result = result.replace('```json', '').replace('```', '').strip()
        elif result.startswith('```'):
            result = result.replace('```', '').strip()
        
        # Parse JSON response
        try:
            json_result = json.loads(result)
            # Extract the normalized value from the JSON
            if isinstance(json_result, dict):
                # Should have the affiliation as key
                return json_result.get(affiliation, list(json_result.values())[0] if json_result else result)
            else:
                return result
        except json.JSONDecodeError:
            # If not valid JSON, return as-is (fallback)
            return result
        
    except Exception as e:
        print(f"  Error processing '{affiliation}': {e}")
        return "ERROR"


def load_existing_gold_set():
    """Load existing gold_set.csv if it exists and has data"""
    if not os.path.exists(output_file):
        return None
    
    try:
        validated = []
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                validated.append(row)
        
        if len(validated) == 0:
            return None
        
        return validated
    except Exception as e:
        print(f"Warning: Could not read existing gold_set.csv: {e}")
        return None


def print_statistics(validated):
    N = len(validated)
    r = sum(1 for v in validated if v['label'] == 'r')
    w = N - r
    accuracy = (r / N) * 100 if N > 0 else 0
    error_rate = (w / N) * 100 if N > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL VALIDATION")
    print(f"{'='*70}")
    print(f"N = Total number of manually evaluated entries: {N}")
    print(f"R = Number of correct normalizations (label = r): {r}")
    print(f"W = Number of incorrect normalizations (label = w): {w}")
    
    print(f"\n{'='*70}")
    print(f"1. ACCURACY")
    print(f"{'='*70}")
    print(f"Proportion of entries the LLM normalized correctly:")
    print(f"Accuracy = R / N = {r} / {N} = {accuracy:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"2. ERROR RATE")
    print(f"{'='*70}")
    print(f"Proportion of incorrect normalizations:")
    print(f"Error Rate = W / N = {w} / {N} = {error_rate:.2f}%")
    
    # Consistency Check
    correct_cases = [v for v in validated if v['label'] == 'r']
    if len(correct_cases) >= 20:
        print(f"\n{'='*70}")
        print(f"CONSISTENCY CHECK")
        print(f"{'='*70}")
        
        # Check if API key is available for consistency check
        if not os.getenv("OPENAI_API_KEY"):
            print(f"Skipping consistency check: OPENAI_API_KEY not set")
            return
        
        print(f"Running consistency check on 20 correct samples...")
        print(f"Testing each sample 3 times to measure variance...\n")
        
        import random
        sample_cases = random.sample(correct_cases, min(20, len(correct_cases)))
        
        consistency_results = []
        inconsistent_count = 0
        total_variance = 0
        
        for idx, case in enumerate(sample_cases, 1):
            rfc_id = case['rfc_id']
            original = case['original_affiliation']
            human_norm = case['human_normalized']
            
            print(f"[{idx}/20] RFC: {rfc_id} - Testing: {original}")
            
            # Run 3 times with temperature=0 for deterministic results
            runs = []
            for run in range(3):
                result = normalize_affiliation(original, temperature=0)
                runs.append(result)
                time.sleep(0.5)
            
            # Check consistency
            all_same = all(r == runs[0] for r in runs)
            consistent = "✓ Consistent" if all_same else "✗ Inconsistent"
            
            # Calculate variance: number of unique outputs
            unique_outputs = len(set(runs))
            variance = unique_outputs - 1  # 0 = no variance, 1 = 2 unique, 2 = 3 unique
            total_variance += variance
            
            if not all_same:
                inconsistent_count += 1
            
            consistency_results.append({
                'rfc_id': rfc_id,
                'original_affiliation': original,
                'human_normalized': human_norm,
                'run_1': runs[0],
                'run_2': runs[1],
                'run_3': runs[2],
                'unique_outputs': unique_outputs,
                'variance': variance,
                'consistent': 'Yes' if all_same else 'No'
            })
            
            print(f"  Run 1: {runs[0]}")
            print(f"  Run 2: {runs[1]}")
            print(f"  Run 3: {runs[2]}")
            print(f"  Unique outputs: {unique_outputs}, Variance: {variance}")
            print(f"  {consistent}\n")
        
        # Calculate statistics
        consistency_rate = ((20 - inconsistent_count) / 20) * 100
        avg_variance = total_variance / 20
        
        print(f"{'='*70}")
        print(f"CONSISTENCY RESULTS:")
        print(f"{'='*70}")
        print(f"Total samples tested: 20")
        print(f"Consistent outputs (variance = 0): {20 - inconsistent_count}")
        print(f"Inconsistent outputs (variance > 0): {inconsistent_count}")
        print(f"Consistency rate: {consistency_rate:.1f}%")
        print(f"\nVariance Metrics:")
        print(f"  Total variance: {total_variance}")
        print(f"  Average variance per sample: {avg_variance:.2f}")
        print(f"  (Variance = number of unique outputs - 1)")
        print(f"  (0 = all same, 1 = 2 different, 2 = all different)")
        
        # Save consistency check results
        consistency_file = os.path.join(curr_dir, "consistency_check.csv")
        try:
            with open(consistency_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['rfc_id', 'original_affiliation', 'human_normalized', 'run_1', 'run_2', 'run_3', 'unique_outputs', 'variance', 'consistent']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(consistency_results)
            print(f"\nConsistency check results saved to: {consistency_file}")
        except Exception as e:
            print(f"ERROR saving consistency check results: {e}")
    else:
        print(f"\n{'='*70}")
        print(f"CONSISTENCY CHECK")
        print(f"{'='*70}")
        print(f"Not enough correct samples for consistency check (need 20, have {len(correct_cases)})")


def validate_normalizations():
    # Check if gold_set.csv already exists and has data
    existing_data = load_existing_gold_set()
    if existing_data is not None:
        print(f"\n{'='*70}")
        print(f"EXISTING GOLD SET FOUND")
        print(f"{'='*70}\n")
        print(f"Found {len(existing_data)} entries in {output_file}")
        print(f"Proceeding directly to statistics and analysis...\n")
        print_statistics(existing_data)
        return
    
    # If no existing data, proceed with validation
    print(f"NO EXISTING GOLD SET FOUND")
    print(f"Proceeding with human validation...\n")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Read raw affiliations and validate columns
    affiliations = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            # Validate required columns
            if not fieldnames or 'rfc_id' not in fieldnames or 'original_affiliation' not in fieldnames:
                print("ERROR: CSV must have 'rfc_id' and 'original_affiliation' columns")
                print(f"Found columns: {fieldnames}")
                return
            
            for row in reader:
                affiliations.append({
                    'rfc_id': row['rfc_id'],
                    'original_affiliation': row['original_affiliation']
                })
    except Exception as e:
        print(f"ERROR reading input file: {e}")
        return
    
    if not affiliations:
        print("ERROR: No affiliations found in input file")
        return
    
    print(f"\n{'='*70}")
    print(f"AFFILIATION NORMALIZATION & VALIDATION")
    print(f"{'='*70}")
    print(f"Total affiliations: {len(affiliations)}")
    print(f"Model: gpt-4.1")
    
    # Initialize validation list
    validated = []
    
    response = input(f"\nProceed with GPT-4.1 normalization and validation? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print(f"\n{'='*70}")
    print(f"VALIDATION INSTRUCTIONS:")
    print(f"  - Review LLM's normalization")
    print(f"  - Press ENTER if LLM is correct (label = r)")
    print(f"  - Type correct normalization if LLM is wrong (label = w)")
    print(f"  - Type 'quit' to save and exit")
    print(f"{'='*70}\n")
    
    # Process each affiliation
    for i in range(len(affiliations)):
        rfc_id = affiliations[i]['rfc_id']
        original = affiliations[i]['original_affiliation']
        
        print(f"\n[{i+1}/{len(affiliations)}] " + "="*50)
        print(f"RFC: {rfc_id}")
        print(f"Processing: {original}")
        
        # Get LLM normalization
        print("Getting LLM normalization...")
        llm_normalized = normalize_affiliation(original, temperature=0)
        
        if llm_normalized == "ERROR":
            print("ERROR: Failed to get LLM normalization. Skipping this entry.")
            response = input("Press ENTER to continue or 'quit' to exit: ")
            if response.lower() == 'quit':
                break
            continue
        
        print(f"\nOriginal:       {original}")
        print(f"LLM Normalized: {llm_normalized}")
        print("-"*60)
        
        # Get human validation
        human_input = input("Correct normalization (ENTER if LLM correct, or type correction): ").strip()
        
        # Handle quit
        if human_input.lower() == 'quit':
            print(f"\nSaving progress... Validated {len(validated)} out of {len(affiliations)}.")
            break
        
        # Determine correct normalization and label
        if not human_input:
            # User pressed ENTER - LLM is correct
            human_normalized = llm_normalized
            label = 'r'
            print("✓ LLM correct (label = r)")
        else:
            # User typed something - LLM is wrong
            human_normalized = human_input
            label = 'w'
            print("✗ LLM incorrect (label = w)")
        
        # Add to validated list
        validated.append({
            'rfc_id': rfc_id,
            'original_affiliation': original,
            'llm_normalized': llm_normalized,
            'human_normalized': human_normalized,
            'label': label
        })
        
        # Save after each validation
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['rfc_id', 'original_affiliation', 'llm_normalized', 'human_normalized', 'label']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(validated)
        except Exception as e:
            print(f"ERROR saving output file: {e}")
            return
        
        # Small delay to avoid rate limits
        if i < len(affiliations) - 1:
            time.sleep(0.3)
    
    # Final summary and statistics
    if validated:
        print_statistics(validated)
    
    print(f"\n{'='*70}")
    print(f"Gold set saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    validate_normalizations()