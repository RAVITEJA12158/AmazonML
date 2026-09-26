def missingness_features(s1: dict, cand: dict) -> dict:
    return {
        "name_missing_s1": float(s1.get("name_missing", False)),
        "name_missing_candidate": float(cand.get("name_missing", False)),
        "address_missing_s1": float(s1.get("address_missing", False)),
        "address_missing_candidate": float(cand.get("address_missing", False)),
        "postal_missing": float(s1.get("postal_missing", False) or cand.get("postal_missing", False)),
        "city_missing": float(s1.get("city_missing", False) or cand.get("city_missing", False)),
        "state_missing": float(s1.get("state_missing", False) or cand.get("state_missing", False)),
        "street_number_missing": float(s1.get("street_number_missing", False) or cand.get("street_number_missing", False)),
        "address_available": float(not (s1.get("address_missing", False) or cand.get("address_missing", False))),
    }
