import os
import csv
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

curr_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(curr_dir, "address_raw.csv")
output_file = os.path.join(curr_dir, "address_gold_set.csv")
error_file = os.path.join(curr_dir, "address_error_samples.csv")
consistency_file = os.path.join(curr_dir, "address_consistency_check.csv")

SYSTEM_PROMPT = """Which country and continent is this address located in?
Simply return a JSON object with two fields: "country" and "continent".
Make sure the names are in consistent format.
For example, dont use U.S for one address and United States of America for other.
Write complete names only always.

Return ONLY the JSON object in this format:
{"country": "country name", "continent": "continent name"}"""


def normalize_address(address, model="gpt-4.1", temperature=0):
    try:
        user_prompt = address
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        
        if result.startswith('```json'):
            result = result.replace('```json', '').replace('```', '').strip()
        elif result.startswith('```'):
            result = result.replace('```', '').strip()
        
        try:
            json_result = json.loads(result)
            country = json_result.get('country', 'Unknown')
            continent = json_result.get('continent', 'Unknown')
            return country, continent
        except json.JSONDecodeError:
            print(f"  Warning: Could not parse JSON response: {result}")
            return "ERROR", "ERROR"
        
    except Exception as e:
        print(f"  Error processing '{address}': {e}")
        return "ERROR", "ERROR"


def load_existing_gold_set():
    """Load existing address_gold_set.csv if it exists and has data"""
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
        print(f"Warning: Could not read existing address_gold_set.csv: {e}")
        return None


def save_error_samples(validated):
    """Save ALL incorrect normalizations to address_error_samples.csv"""
    error_cases = [v for v in validated if v['label'] == 'w']
    
    if not error_cases:
        print(f"\nNo error cases to save (all normalizations were correct!)")
        return
    
    try:
        with open(error_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['rfc_id', 'original_address', 'llm_normalized_country', 
                        'llm_normalized_continent', 'human_normalized_country', 
                        'human_normalized_continent']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(error_cases)
        
        print(f"Error samples saved to: {error_file}")
        print(f"Total error cases: {len(error_cases)}")
    except Exception as e:
        print(f"ERROR saving error samples: {e}")


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
    
    save_error_samples(validated)
    
    # Consistency Check
    correct_cases = [v for v in validated if v['label'] == 'r']
    if len(correct_cases) >= 20:
        print(f"\n{'='*70}")
        print(f"CONSISTENCY CHECK")
        print(f"{'='*70}")
        
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
            original = case['original_address']
            human_country = case['human_normalized_country']
            human_continent = case['human_normalized_continent']
            
            print(f"[{idx}/20] RFC: {rfc_id} - Testing: {original}")
            
            runs = []
            for run in range(3):
                country, continent = normalize_address(original, temperature=0)
                runs.append({'country': country, 'continent': continent})
                time.sleep(0.5)
            
            all_same = all(
                r['country'] == runs[0]['country'] and r['continent'] == runs[0]['continent']
                for r in runs
            )
            consistent = "✓ Consistent" if all_same else "✗ Inconsistent"
            
            # Calculate variance: number of unique outputs
            unique_outputs = len(set((r['country'], r['continent']) for r in runs))
            variance = unique_outputs - 1
            total_variance += variance
            
            if not all_same:
                inconsistent_count += 1
            
            consistency_results.append({
                'rfc_id': rfc_id,
                'original_address': original,
                'human_country': human_country,
                'human_continent': human_continent,
                'run_1_country': runs[0]['country'],
                'run_1_continent': runs[0]['continent'],
                'run_2_country': runs[1]['country'],
                'run_2_continent': runs[1]['continent'],
                'run_3_country': runs[2]['country'],
                'run_3_continent': runs[2]['continent'],
                'unique_outputs': unique_outputs,
                'variance': variance,
                'consistent': 'Yes' if all_same else 'No'
            })
            
            print(f"  Run 1: {runs[0]['country']}, {runs[0]['continent']}")
            print(f"  Run 2: {runs[1]['country']}, {runs[1]['continent']}")
            print(f"  Run 3: {runs[2]['country']}, {runs[2]['continent']}")
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
        try:
            with open(consistency_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['rfc_id', 'original_address', 'human_country', 'human_continent',
                            'run_1_country', 'run_1_continent', 'run_2_country', 'run_2_continent',
                            'run_3_country', 'run_3_continent', 'unique_outputs', 'variance', 'consistent']
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
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    addresses = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if not fieldnames or 'rfc_id' not in fieldnames or 'original_address' not in fieldnames:
                print("ERROR: CSV must have 'rfc_id' and 'original_address' columns")
                print(f"Found columns: {fieldnames}")
                return
            
            for row in reader:
                addresses.append({
                    'rfc_id': row['rfc_id'],
                    'original_address': row['original_address']
                })
    except Exception as e:
        print(f"ERROR reading input file: {e}")
        return
    
    if not addresses:
        print("ERROR: No addresses found in input file")
        return
    
    print(f"\n{'='*70}")
    print(f"ADDRESS NORMALIZATION & VALIDATION")
    print(f"{'='*70}")
    print(f"Total addresses: {len(addresses)}")
    print(f"Model: gpt-4.1")
    
    validated = []
    
    response = input(f"\nProceed with GPT-4.1 normalization and validation? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print(f"\n{'='*70}")
    print(f"VALIDATION INSTRUCTIONS:")
    print(f"  - Review LLM's country and continent extraction")
    print(f"  - Enter correct country and continent (comma-separated)")
    print(f"  - Press ENTER if LLM is completely correct (label = r)")
    print(f"  - Type 'quit' to save and exit")
    print(f"  - Label = r ONLY if BOTH country AND continent are correct")
    print(f"{'='*70}\n")
    
    for i in range(len(addresses)):
        rfc_id = addresses[i]['rfc_id']
        original = addresses[i]['original_address']
        
        print(f"\n[{i+1}/{len(addresses)}] " + "="*50)
        print(f"RFC: {rfc_id}")
        print(f"Processing: {original}")
        
        print("Getting LLM normalization...")
        llm_country, llm_continent = normalize_address(original, temperature=0)
        
        if llm_country == "ERROR" or llm_continent == "ERROR":
            print("ERROR: Failed to get LLM normalization. Skipping this entry.")
            response = input("Press ENTER to continue or 'quit' to exit: ")
            if response.lower() == 'quit':
                break
            continue
        
        print(f"\nOriginal:        {original}")
        print(f"LLM Country:     {llm_country}")
        print(f"LLM Continent:   {llm_continent}")
        print("-"*60)
        
        human_input = input("Correct normalization (ENTER if correct, or type 'country, continent'): ").strip()
        
        if human_input.lower() == 'quit':
            print(f"\nSaving progress... Validated {len(validated)} out of {len(addresses)}.")
            break
        
        if not human_input:
            # User pressed ENTER - LLM is correct for both
            human_country = llm_country
            human_continent = llm_continent
            label = 'r'
            print("✓ LLM correct (label = r)")
        else:
            parts = [p.strip() for p in human_input.split(',')]
            if len(parts) != 2:
                print("ERROR: Please enter 'country, continent' separated by comma")
                i -= 1  
                continue
            
            human_country = parts[0]
            human_continent = parts[1]
            
            if human_country == llm_country and human_continent == llm_continent:
                label = 'r'
                print("✓ LLM correct (label = r)")
            else:
                label = 'w'
                print("✗ LLM incorrect (label = w)")
        
        validated.append({
            'rfc_id': rfc_id,
            'original_address': original,
            'llm_normalized_country': llm_country,
            'llm_normalized_continent': llm_continent,
            'human_normalized_country': human_country,
            'human_normalized_continent': human_continent,
            'label': label
        })
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['rfc_id', 'original_address', 'llm_normalized_country', 
                            'llm_normalized_continent', 'human_normalized_country', 
                            'human_normalized_continent', 'label']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(validated)
        except Exception as e:
            print(f"ERROR saving output file: {e}")
            return
        
        if i < len(addresses) - 1:
            time.sleep(0.3)
    
    if validated:
        print_statistics(validated)
    
    print(f"\n{'='*70}")
    print(f"OUTPUT FILES GENERATED")
    print(f"{'='*70}")
    print(f"1. Gold set: {output_file}")
    print(f"2. Error samples: {error_file}")
    print(f"3. Consistency check: {consistency_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    validate_normalizations()