# decision.py
from config import ROAD_TYPE_CONFIG

def rule_decision(score, road_type):
    cfg = ROAD_TYPE_CONFIG[road_type]
    adjusted_score = score * cfg["score_multiplier"]

    if adjusted_score >= cfg["heavy_threshold"]:
        return "Prioritize this lane (Green = 60s)"
    elif adjusted_score >= cfg["moderate_threshold"]:
        return "Increase green time by 10s"
    else:
        return "Normal signal timing (Green = 30s)"


def final_decision(ml_suggestion, rule_suggestion):
    if ml_suggestion == rule_suggestion:
        return ml_suggestion, "High confidence (ML + Rules agree)"
    else:
        return rule_suggestion, "Medium confidence (Rule override)"
