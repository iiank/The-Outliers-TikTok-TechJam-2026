import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

"""
UI helpers and data loaders for the Streamlit dashboard.
"""

def find_results_path() -> Optional[Path]:
    """
    Locates root/results.json.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        candidate = parent / "results.json"
        if candidate.is_file():
            return candidate
    return None


def load_evaluation_data(custom_path: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads results.json from default root or custom specified path.
    Returns (data_dict, error_message).
    """
    target_path = Path(custom_path) if custom_path else find_results_path()

    if not target_path or not target_path.exists():
        return None, f"results.json not found at {target_path or 'root directory'}."

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except Exception as e:
        return None, f"Error reading results.json: {str(e)}"


def parse_scenario_metrics(scenario_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Transforms scenario_metrics dict into Pandas DataFrame."""
    rows = []
    for scenario_name, metrics in scenario_data.items():
        rows.append({
            "Scenario Type": scenario_name,
            "Sample Count": metrics.get("sample_count", 0),
            "Hit Rate @ 10": metrics.get("hit_rate_at_10", 0.0),
            "MRR": metrics.get("mrr", 0.0),
            "MTTC (Turns)": metrics.get("mttc", 0.0),
        })
    df = pd.DataFrame(rows)
    return df


def parse_sessions(sessions_list: list) -> pd.DataFrame:
    """Transforms raw sessions list into DataFrame."""
    df = pd.DataFrame(sessions_list)
    if "hit" in df.columns:
        df["hit"] = df["hit"].astype(bool)
    if "first_hit_turn" in df.columns:
        df["first_hit_turn"] = df["first_hit_turn"].fillna("-")
    if "best_rank" in df.columns:
        df["best_rank"] = df["best_rank"].fillna("-")
    return df