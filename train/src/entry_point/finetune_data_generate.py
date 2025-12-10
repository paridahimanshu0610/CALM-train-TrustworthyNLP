import numpy as np
import json
import re
import copy
import random

def create_age_buckets(ages):
    """
    Create age buckets based on quantiles.
    Input: list of ages (continuous).
    Output: dict with bucket names and representative ages.
    """
    ages = np.array(ages)
    
    buckets = {
        "young":   int(np.percentile(ages, 20)),
        "middle":  int(np.percentile(ages, 50)),
        "senior":  int(np.percentile(ages, 80))
    }
    return buckets


def generate_counterfactual_ages(original_age, buckets):
    """
    Given an original age and bucket representatives, 
    return counterfactual ages taken from *other buckets*.
    """
    # Determine which bucket original_age belongs to
    young, middle, senior = buckets["young"], buckets["middle"], buckets["senior"]
    
    if original_age <= young:
        original_bucket = "young"
    elif original_age <= middle:
        original_bucket = "middle"
    elif original_age <= senior:
        original_bucket = "senior"
    else:
        original_bucket = "old"
    
    # Return all representative bucket ages except the one original belongs to
    counterfactual_ages = []
    
    for name, rep_age in buckets.items():
        if name != original_bucket:
            counterfactual_ages.append(rep_age)
    
    return counterfactual_ages

# ages = [22, 25, 30, 35, 40, 45, 52, 60, 67, 75]

# buckets = create_age_buckets(ages)
# print("Buckets:", buckets)

# original_age = 27
# cf_ages = generate_counterfactual_ages(original_age, buckets)
# print("Counterfactual ages:", cf_ages)


IDENTIFIER_MAP = {
    "cc_Fraud": "The client is a female, the state number is 25, the number of cards is 1, the credit balance is 7000, the number of transactions is 16, the number of international transactions is 0, the credit limit is 6.",
    "german": "The client has a stable income, no previous debts, and owns a property.",
    "travel_insurance": ('The insurance company has attributes: Agency: CBH, Agency Type: '
                 'Travel Agency, Distribution Chanel: Offline, Product Name: Comprehensive '
                 'Plan, Duration: 186, Destination: MALAYSIA, Net Sales: -29, Commision: 9.57, Age: 81.')  
}

AGE_IDENTIFIER_MAP = {
    "german": "The state of Age in years is ",
    "travel_insurance": "Age: " 
}

def extract_ages_from_json(json_path, dataset_name):
    IDENTIFIER = IDENTIFIER_MAP[dataset_name]
    age_identifier = AGE_IDENTIFIER_MAP[dataset_name]
    ages = []

    # Load entire JSON file
    data_list = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data_list.append(json.loads(line))

    # Scan each record
    for record in data_list:
        conversations = record.get("conversations", [])
        
        # Find the human message
        for convo in conversations:
            if convo.get("from") != "human":
                continue

            value = convo.get("value", "")

            # Check if this is a record of interest
            if IDENTIFIER not in value:
                continue  # Skip this record

            # Extract the text after '\nText:'
            text_match = re.search(r"Text:\s*'(.*?)'", value, flags=re.DOTALL)
            if not text_match:
                continue

            text_block = text_match.group(1)

            age_match = re.search(rf"{age_identifier}(\d+)", text_block)
            if age_match:
                ages.append(int(age_match.group(1)))

    return ages

train_filename = '/Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/train/data/CRA-resample-train4w_original.json'
german_ages = extract_ages_from_json(train_filename, "german")
print(f"Are all {len(german_ages)} German age integers: ", all(isinstance(x, int) for x in german_ages))
travel_insurance_ages = extract_ages_from_json(train_filename, "travel_insurance")
print(f"Are all {len(travel_insurance_ages)} Travel Insurance age integers: ", all(isinstance(x, int) for x in travel_insurance_ages))

# Creating age buckets
german_age_bucket = create_age_buckets(german_ages)
print("german_age_bucket: ", german_age_bucket)
travel_insurance_age_bucket = create_age_buckets(travel_insurance_ages)
print("travel_insurance_age_bucket: ", travel_insurance_age_bucket)

def cc_toggle_gender_in_text(input_dict):
    """
    Takes the full dictionary, finds the human message in conversations,
    detects the line following '\nText:', and flips 'male' <-> 'female' 
    ONLY inside that Text block. Returns a modified copy of the dictionary.
    """
    
    # Create a deep copy so original is untouched
    data = copy.deepcopy(input_dict)
    
    # Find human message
    for convo in data.get("conversations", []):
        if convo.get("from") == "human":
            value = convo.get("value", "")

            # Locate the block starting after '\nText:'
            # Example target: Text: 'The client is a female, ...'
            match = re.search(r"(Text:\s*')(.*?)(')", value, flags=re.DOTALL)
            if not match:
                return data  # Nothing to modify
            
            prefix, text_block, suffix = match.groups()

            # Toggle male ↔ female safely using placeholder
            updated_text = (
                text_block
                .replace("female", "__TEMP_FEMALE__")
                .replace("male", "female")
                .replace("__TEMP_FEMALE__", "male")
            )

            # Rebuild updated value
            updated_value = value.replace(prefix + text_block + suffix,
                                          prefix + updated_text + suffix)
            
            convo["value"] = updated_value

    return data

def german_compute_new_age(current_age: int) -> int:
    """
    Replace this logic with whatever transformation you want for age.
    For now, as an example, return current_age + 1.
    """
    return current_age + 1


def german_transform_text_rules(text_block, age_bucket = german_age_bucket):
    """
    Applies rule-based transformations to the Text block:
      - Flip male <-> female
      - Flip foreign worker yes <-> no
      - Replace the age with a transformed age n0 = compute_new_age(current_age)
    """

    # 1. ---- Flip male <-> female ----
    text = (text_block
        .replace("male", "__TEMP_MALE__")
        .replace("female", "male")
        .replace("__TEMP_MALE__", "female")
    )

    # 2. ---- Flip foreign worker yes <-> no ----
    text = (text
        .replace("foreign worker is yes", "foreign worker is __TEMP_FW__")
        .replace("foreign worker is no", "foreign worker is yes")
        .replace("foreign worker is __TEMP_FW__", "foreign worker is no")
    )

    # 3. ---- Replace Age in years X -> computed n0 ----
    random.seed(1234)
    age_match = re.search(r"Age in years is (\d+)", text)
    if age_match:
        current_age = int(age_match.group(1))
        new_age = random.choice(generate_counterfactual_ages(current_age, age_bucket))
        text = re.sub(
            r"Age in years is \d+",
            f"Age in years is {new_age}",
            text
        )

    return text


def german_transform_dictionary(input_dict, age_bucket = german_age_bucket):
    """
    Finds the human message, extracts the \\nText block, applies transformations,
    and returns the full modified dictionary.
    """

    data = copy.deepcopy(input_dict)

    for convo in data.get("conversations", []):
        if convo.get("from") == "human":
            value = convo.get("value", "")

            # Extract the content inside: Text: ' ... '
            match = re.search(r"(Text:\s*')(.*?)(')", value, flags=re.DOTALL)
            if not match:
                return data

            prefix, text_block, suffix = match.groups()

            # Apply the rule-based transformation
            updated_text = german_transform_text_rules(text_block, age_bucket = german_age_bucket)

            # Update the entire message
            updated_value = value.replace(
                prefix + text_block + suffix,
                prefix + updated_text + suffix
            )

            convo["value"] = updated_value

    return data

def trin_compute_new_age(current_age: int) -> int:
    """
    Replace this with your true age transformation logic.
    Example: return current_age + 5
    """
    return current_age + 5


def trin_transform_text_age(text_block, age_bucket = travel_insurance_age_bucket):
    """
    Finds 'Age: N' inside the Text block, computes a new age using compute_new_age(),
    and replaces the old age with the computed one.
    """
    random.seed(1234)
    age_match = re.search(r"Age:\s*(\d+)", text_block)
    if age_match:
        current_age = int(age_match.group(1))
        new_age = random.choice(generate_counterfactual_ages(current_age, age_bucket))

        # Replace only the age number
        text_block = re.sub(
            r"Age:\s*\d+",
            f"Age: {new_age}",
            text_block
        )

    return text_block


def trin_transform_dictionary_age_only(input_dict, age_bucket = travel_insurance_age_bucket):
    """
    Finds the human message, extracts the \\nText block,
    applies only the Age transformation, and returns the full dictionary.
    """

    data = copy.deepcopy(input_dict)

    for convo in data.get("conversations", []):
        if convo.get("from") == "human":
            value = convo.get("value", "")

            # Extract content inside: Text: ' ... '
            match = re.search(r"(Text:\s*')(.*?)(')", value, flags=re.DOTALL)
            if not match:
                return data

            prefix, text_block, suffix = match.groups()

            # Apply only the age update
            updated_text = trin_transform_text_age(text_block, age_bucket = travel_insurance_age_bucket)

            # Rebuild the updated message
            updated_value = value.replace(
                prefix + text_block + suffix,
                prefix + updated_text + suffix
            )
            convo["value"] = updated_value

    return data

IDENTIFIER_MAP = {
    "cc_Fraud": "The client is a female, the state number is 25, the number of cards is 1, the credit balance is 7000, the number of transactions is 16, the number of international transactions is 0, the credit limit is 6.",
    "german": "The client has a stable income, no previous debts, and owns a property.",
    "travel_insurance": ('The insurance company has attributes: Agency: CBH, Agency Type: '
                 'Travel Agency, Distribution Chanel: Offline, Product Name: Comprehensive '
                 'Plan, Duration: 186, Destination: MALAYSIA, Net Sales: -29, Commision: 9.57, Age: 81.')  
}

def create_counter_factual_records(json_path, identifier_map = IDENTIFIER_MAP):
    # Load entire JSON file
    data_list = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data_list.append(json.loads(line))

    german_cnt, trin_cnt, ccFraud_cnt = 0, 0, 0
    german_records, trin_records, ccFraud_records = [], [], []
    
    # Scan each record
    for record in data_list:
        conversations = record.get("conversations", [])
        
        # Find the human message
        for convo in conversations:
            if convo.get("from") != "human":
                continue

            value = convo.get("value", "")

            # Check if this is a record of interest
            if IDENTIFIER_MAP["cc_Fraud"] in value:
                ccFraud_records.append(cc_toggle_gender_in_text(record))
                ccFraud_cnt += 1
                break
            elif IDENTIFIER_MAP["german"] in value:
                german_records.append(german_transform_dictionary(record))
                german_cnt += 1
                break
            elif IDENTIFIER_MAP["travel_insurance"] in value:
                trin_records.append(trin_transform_dictionary_age_only(record))
                trin_cnt += 1
                break

    random.seed(1234)
    trin_records = random.sample(trin_records, trin_cnt//2)
    ccFraud_records = random.sample(ccFraud_records, ccFraud_cnt//2)

    data_list.extend(german_records + trin_records + ccFraud_records)

    print(f"Created {ccFraud_cnt//2} records for ccFraud, {german_cnt} records for German, and {trin_cnt//2} for Travel Insurance. \n Total: {german_cnt + trin_cnt//2 + ccFraud_cnt//2}")
    return data_list


def save_as_json_lines(data, file_path):
    """
    data: list of dicts
    file_path: path ending with .json
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

pp = create_counter_factual_records('/Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/train/data/CRA-resample-train4w_original.json')
save_as_json_lines(pp, '/Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/train/data/CRA-resample-train4w.json')