"""Generates a larger, harder synthetic dataset so the pipeline's reported
accuracy actually means something. The previous generator (15 S1 entities,
~60 S2+S3 records) was too small and too easy for macro F0.5 = 1.0 to be
evidence of anything except memorizing a toy set -- with a handful of near-
duplicate distractors and a couple hundred noise rows, a perfect score there
just means "the recall ceiling is trivially 1.0", not "the model is good".

This version scales up combinatorially and adds noise patterns that are
specifically hard for a naive name/address matcher:
  - ~260 base businesses built from word-pool combinations (not 15 fixed
    strings), across 8 city/country combinations
  - legal-suffix variants, abbreviation swaps, transliteration-style
    character substitutions, token drops, digit transpositions in street
    numbers and postal codes
  - DBA / trade-name style renames (same business, materially different name)
  - near-duplicate distractor clusters: same city, same street, similar
    name, DIFFERENT business (the classic false-positive trap)
  - postal-code collisions across unrelated businesses in dense areas
  - missing-field patterns (address, postal, city) kept as true nulls, not
    empty strings, so missingness features stay meaningful
  - ~1,400 unrelated "noise" records so blocking has real work to do
  - multi-source support (same true match appearing in S2 and S3) and
    genuine singletons (~28% of S1 entities have no true match at all)

Run: python scripts/00_generate_sample_data.py
"""
import os
import sys
import random
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.utils.seed import set_seed
from src.utils.io import load_config, save_table

INDUSTRY_WORDS = [
    "Technologies", "Pharma", "Traders", "Bakery", "Electronics",
    "Precision Instruments", "Auto Parts", "Foods", "Textiles", "Consulting",
    "Steel Works", "Bookstore", "Digital Solutions", "Furniture", "Imports",
    "Logistics", "Ventures", "Holdings", "Services", "Chemicals",
    "Engineering", "Apparel", "Motors", "Realty", "Hospitality",
    "Biosciences", "Plastics", "Packaging", "Metals", "Agro Products",
]
PREFIX_WORDS = [
    "ABC", "Shree Balaji", "Global", "City", "Sunrise", "Raviteja", "Metro",
    "Green Valley", "Silver Star", "Oceanview", "Bharat", "Maple Leaf",
    "Nova", "Lakeside", "Golden Gate", "Everest", "Crescent", "Pinnacle",
    "Northstar", "Bluewater", "Redwood", "Summit", "Harbor", "Meridian",
    "Vertex", "Crown", "Horizon", "Cascade", "Ironwood", "Riverside",
]
LEGAL_SUFFIX_VARIANTS = ["", " Pvt Ltd", " Private Limited", " Corp", " Corporation", " Inc", " Ltd", " LLC", " & Co"]

CITIES = [
    ("US", "California", "San Francisco", "94105"),
    ("US", "New York", "New York", "10001"),
    ("US", "Texas", "Austin", "73301"),
    ("India", "Karnataka", "Bengaluru", "560001"),
    ("India", "Maharashtra", "Mumbai", "400001"),
    ("India", "Telangana", "Hyderabad", "500001"),
    ("France", "Ile-de-France", "Paris", "75001"),
    ("Germany", "Bavaria", "Munich", "80331"),
]
STREETS = [
    "Main Street", "MG Road", "Park Avenue", "Rue de Rivoli", "Church Street",
    "Brigade Road", "5th Avenue", "Oak Lane", "Linden Strasse", "Residency Road",
    "Commerce Street", "Station Road", "King Street", "Gandhi Nagar", "Elm Drive",
]

CHAR_SUBS = [("i", "y"), ("ph", "f"), ("c", "k"), ("z", "s"), ("v", "w"), ("0", "o")]
ABBREV_SUBS = [("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Technologies", "Tech"),
               ("Corporation", "Corp"), ("and", "&"), ("Precision Instruments", "Precision Instr")]


def make_base_name(rng: random.Random) -> str:
    return f"{rng.choice(PREFIX_WORDS)} {rng.choice(INDUSTRY_WORDS)}"


def transliterate(text: str, rng: random.Random, prob: float = 0.5) -> str:
    if rng.random() > prob:
        return text
    a, b = rng.choice(CHAR_SUBS)
    return text.replace(a, b) if a in text else text


def abbreviate(text: str, rng: random.Random, prob: float = 0.4) -> str:
    if rng.random() > prob:
        return text
    a, b = rng.choice(ABBREV_SUBS)
    return text.replace(a, b) if a in text else text


def noisy_name(name: str, rng: random.Random, dba_prob: float = 0.08) -> str:
    if rng.random() < dba_prob:
        prefix = name.split()[0]
        name = f"{prefix} {rng.choice(INDUSTRY_WORDS)}"
    name = abbreviate(name, rng)
    name = transliterate(name, rng, prob=0.25)
    if rng.random() < 0.1:
        tokens = name.split()
        if len(tokens) > 2:
            tokens.pop(rng.randrange(len(tokens)))
            name = " ".join(tokens)
    return name + rng.choice(LEGAL_SUFFIX_VARIANTS)


def transpose_digits(s: str, rng: random.Random) -> str:
    if not s or len(s) < 2:
        return s
    i = rng.randrange(len(s) - 1)
    chars = list(s)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def noisy_address(street_number, street, city_tuple, rng, drop_prob=0.2, corrupt_postal_prob=0.1):
    country, state, city, postal = city_tuple
    parts = []
    if rng.random() > drop_prob:
        num = str(street_number)
        if rng.random() < 0.08:
            num = transpose_digits(num, rng)
        parts.append(num)
    street_out = abbreviate(street, rng, prob=0.35)
    parts.append(street_out)
    addr = " ".join(parts)

    out_postal = postal
    if rng.random() < drop_prob:
        out_postal = None
    elif rng.random() < corrupt_postal_prob:
        out_postal = transpose_digits(postal, rng)

    out_city = None if rng.random() < drop_prob * 0.4 else city
    return addr, out_postal, out_city, state, country


def main():
    set_seed(42)
    rng = random.Random(42)
    cfg = load_config("config.yaml")
    raw_dir = cfg["paths"]["raw_dir"]

    s1_rows, s2_rows, s3_rows, label_rows = [], [], [], []
    used_names = set()
    n_entities = 260

    for i in range(n_entities):
        entity_id = f"S1_{i:04d}"
        for _ in range(20):
            base_name = make_base_name(rng)
            if base_name not in used_names:
                used_names.add(base_name)
                break

        city_tuple = rng.choice(CITIES)
        street = rng.choice(STREETS)
        street_number = rng.randint(1, 900)

        addr, postal, city, state, country = noisy_address(street_number, street, city_tuple, rng, drop_prob=0.08)
        s1_rows.append({
            "source1_entity_id": entity_id, "name": base_name, "address": addr,
            "country": country, "state": state, "city": city, "postal_code": postal,
            "street_number": street_number, "street_name": street, "phone": None, "email": None,
        })

        is_singleton = rng.random() < 0.28
        n_matches = 0 if is_singleton else rng.choice([1, 1, 1, 2, 2, 3])

        for m in range(n_matches):
            for source_tag, rows in (("S2", s2_rows), ("S3", s3_rows)):
                if rng.random() < 0.72:
                    rec_id = f"{source_tag}_{entity_id}_{m}"
                    n_name = noisy_name(base_name, rng)
                    n_addr, n_postal, n_city, n_state, n_country = noisy_address(
                        street_number, street, city_tuple, rng, drop_prob=0.22, corrupt_postal_prob=0.12)
                    rows.append({
                        "record_id": rec_id, "name": n_name, "address": n_addr,
                        "country": n_country, "state": n_state, "city": n_city, "postal_code": n_postal,
                        "street_number": street_number if rng.random() > 0.25 else None,
                        "street_name": street if rng.random() > 0.1 else None,
                        "phone": None, "email": None,
                    })
                    label_rows.append({"source1_entity_id": entity_id, "candidate_source": source_tag,
                                        "candidate_record_id": rec_id, "label": 1})

        if rng.random() < 0.55:
            hn_id = f"S2_HN1_{entity_id}"
            hn_name = base_name.split()[0] + " " + rng.choice([w for w in INDUSTRY_WORDS if w not in base_name])
            hn_addr, hn_postal, hn_city, hn_state, hn_country = noisy_address(
                rng.randint(1, 900), street, city_tuple, rng, drop_prob=0.1)
            s2_rows.append({
                "record_id": hn_id, "name": hn_name, "address": hn_addr,
                "country": hn_country, "state": hn_state, "city": hn_city, "postal_code": hn_postal,
                "street_number": None, "street_name": None, "phone": None, "email": None,
            })
            label_rows.append({"source1_entity_id": entity_id, "candidate_source": "S2",
                                "candidate_record_id": hn_id, "label": 0})

        if rng.random() < 0.35:
            other_city = rng.choice([c for c in CITIES if c != city_tuple])
            hn_id = f"S3_HN2_{entity_id}"
            hn_addr, hn_postal, hn_city, hn_state, hn_country = noisy_address(
                rng.randint(1, 900), rng.choice(STREETS), other_city, rng, drop_prob=0.05)
            s3_rows.append({
                "record_id": hn_id, "name": base_name, "address": hn_addr,
                "country": hn_country, "state": hn_state, "city": hn_city, "postal_code": hn_postal,
                "street_number": None, "street_name": None, "phone": None, "email": None,
            })
            label_rows.append({"source1_entity_id": entity_id, "candidate_source": "S3",
                                "candidate_record_id": hn_id, "label": 0})

        if rng.random() < 0.25:
            hn_id = f"S2_HN3_{entity_id}"
            hn_name = make_base_name(rng)
            s2_rows.append({
                "record_id": hn_id, "name": hn_name,
                "address": f"{rng.randint(1, 900)} {rng.choice(STREETS)}",
                "country": country, "state": state, "city": city, "postal_code": postal,
                "street_number": None, "street_name": None, "phone": None, "email": None,
            })
            label_rows.append({"source1_entity_id": entity_id, "candidate_source": "S2",
                                "candidate_record_id": hn_id, "label": 0})

    n_noise = 1400
    for j in range(n_noise):
        city_tuple = rng.choice(CITIES)
        street = rng.choice(STREETS)
        name = make_base_name(rng)
        addr, postal, city, state, country = noisy_address(rng.randint(1, 900), street, city_tuple, rng)
        target = s2_rows if j % 2 == 0 else s3_rows
        target.append({
            "record_id": f"{'S2' if j % 2 == 0 else 'S3'}_NOISE_{j:05d}",
            "name": name, "address": addr, "country": country, "state": state, "city": city,
            "postal_code": postal, "street_number": None, "street_name": None, "phone": None, "email": None,
        })

    save_table(pd.DataFrame(s1_rows), os.path.join(raw_dir, "source1.tsv"))
    save_table(pd.DataFrame(s2_rows), os.path.join(raw_dir, "source2.tsv"))
    save_table(pd.DataFrame(s3_rows), os.path.join(raw_dir, "source3.tsv"))
    save_table(pd.DataFrame(label_rows), os.path.join(raw_dir, "labels.tsv"))

    n_pos = sum(1 for r in label_rows if r["label"] == 1)
    n_neg = len(label_rows) - n_pos
    print(f"Generated: {len(s1_rows)} S1, {len(s2_rows)} S2, {len(s3_rows)} S3 "
          f"({len(s2_rows) + len(s3_rows)} total candidate-source records)")
    print(f"Labeled pairs: {len(label_rows)} ({n_pos} positive / {n_neg} hard-negative)")
    print(f"Saved to {raw_dir}/")


if __name__ == "__main__":
    main()
