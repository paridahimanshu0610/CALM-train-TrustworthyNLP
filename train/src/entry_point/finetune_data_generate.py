"""
Generate counterfactual training examples for the German Credit dataset
records found inside a larger multi-dataset training file.

Targeted (not uniform) augmentation, per our bias diagnosis:
  - Gender / Foreigner: only flip privileged-group + BAD-labeled records
    (male->female, non-foreign->foreign), keeping the label "bad".
    This targets the FPR gap specifically (leniency bias).
  - Age: flip YOUNG records to OLD, split by label:
      young+good -> old+good   (targets the EOD / false-negative gap)
      young+bad  -> old+bad    (targets the FPR / false-positive gap)
"""

import json
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants: code <-> human-readable text mappings used in the prompt text
# ---------------------------------------------------------------------------

AGE_THRESHOLD = 45

STATUS_MAP = {
    'A91': 'male: divorced or separated',
    'A92': 'female: divorced or separated or married',
    'A93': 'male and single',
    'A94': 'male and married or widowed',
    'A95': 'female and single',
}
EMPLOYMENT_MAP = {
    'A71': 'unemployed',
    'A72': 'smaller than 1 year',
    'A73': 'bigger than 1  smaller than 4 years',
    'A74': 'bigger than 4  smaller than 7 years',
    'A75': 'bigger than 7 years',
}
HOUSING_MAP = {
    'A151': 'rent',
    'A152': 'own',
    'A153': 'for free',
}
GENDER_MAP = {'A91': 'male', 'A93': 'male', 'A94': 'male', 'A92': 'female', 'A95': 'female'}

REVERSE_STATUS_MAP = {v: k for k, v in STATUS_MAP.items()}
REVERSE_EMPLOYMENT_MAP = {v: k for k, v in EMPLOYMENT_MAP.items()}
REVERSE_HOUSING_MAP = {v: k for k, v in HOUSING_MAP.items()}

GERMAN_MARKER = "For instance, 'The client has a stable income, no previous debts, and owns a property.' should be classified as 'good'."

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Part 1: distribution-building + flip helpers
# ---------------------------------------------------------------------------

def _age_group(age):
    return 'young' if age <= AGE_THRESHOLD else 'old'


def _opposite_gender(gender):
    return 'female' if gender == 'male' else 'male'


def build_conditional_distributions_joint(df, group_col):
    dist = {}
    for g, sub in df.groupby(group_col):
        pair_counts = sub.groupby(['employment_since', 'housing']).size()
        pair_probs = (pair_counts / pair_counts.sum()).to_dict()
        dist[g] = {
            'pair_dist': pair_probs,
            'age_pool': sub['age'].to_numpy(),
        }
    return dist


def _sample_pair(pair_dist, rng):
    pairs = list(pair_dist.keys())
    probs = np.array(list(pair_dist.values()))
    probs = probs / probs.sum()
    idx = rng.choice(len(pairs), p=probs)
    return pairs[idx]


def sample_old_attributes(age_cond_dist, rng=RNG):
    """Sample an (age, employment_since, housing) triple from the 'old' group."""
    target_dist = age_cond_dist['old']
    new_age = int(rng.choice(target_dist['age_pool']))
    new_employment_since, new_housing = _sample_pair(target_dist['pair_dist'], rng)
    return {'age': new_age, 'employment_since': new_employment_since, 'housing': new_housing}


def sample_gender_attributes(target_gender, gender_cond_dist, rng=RNG):
    """Sample an (employment_since, housing) pair from the target gender's distribution."""
    target_dist = gender_cond_dist[target_gender]
    new_employment_since, new_housing = _sample_pair(target_dist['pair_dist'], rng)
    return {'gender': target_gender, 'employment_since': new_employment_since, 'housing': new_housing}


# ---------------------------------------------------------------------------
# Part 2: gender <-> personal_status_sex code heuristic
# ---------------------------------------------------------------------------

def flip_personal_status(status_code, df):
    direct_flip = {
        'A93': 'A95',  # male and single -> female and single
        'A95': 'A93',  # female and single -> male and single
        'A91': 'A92',  # male divorced/separated -> female divorced/separated/married
        'A94': 'A92',  # male married/widowed -> female divorced/separated/married
    }
    if status_code in direct_flip:
        return direct_flip[status_code]

    if status_code == 'A92':
        counts = df['personal_status_sex'].value_counts()
        a91, a94 = counts.get('A91', 0), counts.get('A94', 0)
        total = a91 + a94
        p_a91 = a91 / total if total > 0 else 0.5
        return 'A91' if RNG.random() < p_a91 else 'A94'

    raise ValueError(f"Unrecognized personal_status_sex code: {status_code}")


# ---------------------------------------------------------------------------
# Part 3: text parsing / editing helpers
# ---------------------------------------------------------------------------

def is_german_record(human_text):
    return GERMAN_MARKER in human_text


def extract_field_value(text, field_label):
    pattern = re.compile(
        r"The state of\s+" + re.escape(field_label) + r"\s+is\s+(.*?)\.\s*(?:The state of|')",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def replace_field_value(text, field_label, new_value):
    pattern = re.compile(
        r"(The state of\s+" + re.escape(field_label) + r"\s+is\s+)(.*?)(\.\s*(?:The state of|'))",
        re.IGNORECASE,
    )

    def _sub(match):
        return match.group(1) + new_value + match.group(3)

    new_text, n_subs = pattern.subn(_sub, text, count=1)
    if n_subs == 0:
        raise ValueError(f"Could not find field '{field_label}' in text to replace.")
    return new_text


def get_human_message(conversations):
    for turn in conversations:
        if turn.get('from') == 'human':
            return turn
    return None


def get_assistant_label(conversations):
    """Return the assistant's 'good'/'bad' label for this record, lowercased."""
    for turn in conversations:
        if turn.get('from') == 'assistant':
            return turn['value'].strip().lower()
    return None


# ---------------------------------------------------------------------------
# Part 4: parse a German record's relevant fields in one place
# ---------------------------------------------------------------------------

def parse_german_record(record):
    """
    Extract everything we need to decide whether/how to build a counterfactual:
    age, gender (derived from personal_status_sex), foreign-worker status, and label.
    Returns None if any field can't be parsed.
    """
    human_turn = get_human_message(record['conversations'])
    if human_turn is None:
        return None
    text = human_turn['value']
    if not is_german_record(text):
        return None

    age_text = extract_field_value(text, "Age in years")
    status_text = extract_field_value(text, "Personal status and sex")
    foreign_text = extract_field_value(text, "foreign worker")
    label = get_assistant_label(record['conversations'])

    if age_text is None or status_text is None or foreign_text is None or label is None:
        return None

    status_code = REVERSE_STATUS_MAP.get(status_text)
    if status_code is None:
        return None

    return {
        'text': text,
        'age': int(age_text),
        'status_code': status_code,
        'gender': GENDER_MAP.get(status_code),
        'foreign': foreign_text.strip().lower(),
        'label': label,
    }


# ---------------------------------------------------------------------------
# Part 5: build one counterfactual record, given the fields to change
# ---------------------------------------------------------------------------

def build_record_with_text(record, new_text, suffix):
    new_record = json.loads(json.dumps(record))  # deep copy
    new_human_turn = get_human_message(new_record['conversations'])
    new_human_turn['value'] = new_text
    new_record['id'] = f"{record['id']}-{suffix}"
    return new_record


def make_gender_or_foreign_counterfactual(record, parsed, cf_type, gender_cond_dist, df):
    """
    cf_type: 'gender' (male -> female) or 'foreign' (non-foreign -> foreign).
    Only called on records that already passed the privileged+bad-label filter.
    """
    text = parsed['text']

    if cf_type == 'gender':
        target_gender = 'female'  # we only ever flip FROM male (privileged) TO female
        flipped = sample_gender_attributes(target_gender, gender_cond_dist)
        new_status_code = flip_personal_status(parsed['status_code'], df)

        text = replace_field_value(text, "Personal status and sex", STATUS_MAP[new_status_code])
        text = replace_field_value(text, "Present employment since", EMPLOYMENT_MAP[flipped['employment_since']])
        text = replace_field_value(text, "Housing", HOUSING_MAP[flipped['housing']])
        suffix = "gender-cf"

    elif cf_type == 'foreign':
        text = replace_field_value(text, "foreign worker", "yes")  # non-foreign -> foreign
        suffix = "foreign-cf"

    else:
        raise ValueError(f"Unknown cf_type: {cf_type}")

    return build_record_with_text(record, text, suffix)


def make_age_counterfactual(record, parsed, age_cond_dist):
    """
    Only called on YOUNG records (either label). Flips to an 'old' profile,
    keeping the original label unchanged.
    """
    flipped = sample_old_attributes(age_cond_dist)

    text = replace_field_value(parsed['text'], "Age in years", str(flipped['age']))
    text = replace_field_value(text, "Present employment since", EMPLOYMENT_MAP[flipped['employment_since']])
    text = replace_field_value(text, "Housing", HOUSING_MAP[flipped['housing']])

    suffix = f"age-cf-{parsed['label']}"  # e.g. age-cf-good / age-cf-bad, for traceability
    return build_record_with_text(record, text, suffix)


# ---------------------------------------------------------------------------
# Part 6: main entry point
# ---------------------------------------------------------------------------

def generate_counterfactuals(train_data_path, df, output_path=None):
    with open(train_data_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    df = df.copy()
    df['age_group'] = df['age'].apply(_age_group)
    age_cond_dist = build_conditional_distributions_joint(df, 'age_group')
    gender_cond_dist = build_conditional_distributions_joint(df, 'gender')

    counterfactual_records = []
    skipped = 0
    counts = {'gender': 0, 'foreign': 0, 'age_good': 0, 'age_bad': 0}

    for record in records:
        parsed = parse_german_record(record)
        if parsed is None:
            continue  # not a (parseable) German record

        # --- Gender: only privileged group (male) + unfavorable label (bad) ---
        if parsed['gender'] == 'male' and parsed['label'] == 'bad':
            new_record = make_gender_or_foreign_counterfactual(
                record, parsed, 'gender', gender_cond_dist, df
            )
            if new_record:
                counterfactual_records.append(new_record)
                counts['gender'] += 1

        # --- Foreigner: only privileged group (non-foreign) + unfavorable label (bad) ---
        if parsed['foreign'] == 'no' and parsed['label'] == 'bad':
            new_record = make_gender_or_foreign_counterfactual(
                record, parsed, 'foreign', gender_cond_dist, df
            )
            if new_record:
                counterfactual_records.append(new_record)
                counts['foreign'] += 1

        # --- Age: only young records, both labels, kept separate ---
        if _age_group(parsed['age']) == 'young':
            new_record = make_age_counterfactual(record, parsed, age_cond_dist)
            if new_record:
                counterfactual_records.append(new_record)
                counts[f"age_{parsed['label']}"] += 1

    print(f"Generated {len(counterfactual_records)} counterfactual records.")
    print(f"  Gender (male+bad -> female+bad):    {counts['gender']}")
    print(f"  Foreign (non-foreign+bad -> foreign+bad): {counts['foreign']}")
    print(f"  Age, good-labeled (young -> old):   {counts['age_good']}")
    print(f"  Age, bad-labeled  (young -> old):   {counts['age_bad']}")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(counterfactual_records, f, indent=2)
        print(f"Saved to {output_path}")

    return counterfactual_records


if __name__ == "__main__":
    col_names = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
                 'savings', 'employment_since', 'installment_rate', 'personal_status_sex',
                 'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
                 'housing', 'existing_credits', 'job', 'num_liable', 'telephone', 'foreign_worker',
                 'class']

    df = pd.read_csv(
        '/Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/data/original_data/credit_scoring/German/german.data',
        sep=r'\s+', names=col_names
    )
    df['gender'] = df['personal_status_sex'].map(GENDER_MAP)

    TRAIN_DATA_PATH = "train/data/CRA-resample-train4w.json"
    OUTPUT_PATH = "train/data/CRA-german-counterfactuals.json"

    generate_counterfactuals(TRAIN_DATA_PATH, df, output_path=OUTPUT_PATH)