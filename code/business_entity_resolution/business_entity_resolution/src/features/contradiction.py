def contradiction_features(s1: dict, cand: dict) -> dict:
    def _conflict(a, b):
        if a is None or b is None:
            return 0.0  # missing is not a conflict — see missingness.py
        return float(str(a).strip() != str(b).strip())

    return {
        "city_conflict": _conflict(s1.get("city"), cand.get("city")),
        "state_conflict": _conflict(s1.get("state"), cand.get("state")),
        "postal_conflict": _conflict(s1.get("postal_code"), cand.get("postal_code")),
        "street_number_conflict": _conflict(s1.get("street_number"), cand.get("street_number")),
        "country_conflict": _conflict(s1.get("country"), cand.get("country")),
    }
