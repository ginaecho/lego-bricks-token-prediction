"""Read-only index over the measured build corpus and its brick catalog."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Corpus:
    def __init__(self, root: Path):
        self.root = root / "data" / "build_simulations"
        self.points = [_read(path) for path in sorted((self.root / "data_points").glob("*.json"))]
        self.measured = [p for p in self.points if p["status"] == "measured_build_passed"
                         and p["origin"] == "catalog_designed_simulation"]
        self.by_id = {p["id"]: p for p in self.points}
        self.by_group = defaultdict(list)
        for point in self.measured:
            self.by_group[point["split_group"]].append(point)

    def catalog(self) -> dict:
        families = _read(self.root / "basic_functionality_registry.json")
        types = {item["type"]: item for item in _read(self.root / "functionality_registry.json")}
        industries = _read(self.root / "industry_profiles.json")["profiles"]
        return {
            "bricks": [{
                "id": family["basic_functionality_id"], "name": family["basic_functionality"],
                "category": family["category"], "description": family["studio_description"],
                "planned_operations": family["planned_operations"],
                "variants": [{"id": name, "name": types[name]["type_name"],
                              "requirement": types[name]["build_requirement"]} for name in family["types"]],
            } for family in families],
            "industries": [{"id": key, **value} for key, value in industries.items()],
        }

    def matches(self, split_group: str, parts: list[str]) -> list[dict]:
        """Measured builds of the same membership; exact execution order first."""
        def order(point):
            features = point["input_features"]
            return (features.get("parts") or features["types"]) != parts, point["id"]
        return sorted(self.by_group.get(split_group, []), key=order)

    def build_files(self, point_id: str) -> dict:
        directory = self.root / "builds" / point_id
        if not directory.is_dir():
            return {}
        return {path.name: {"bytes": path.stat().st_size, "lines": path.read_text(encoding="utf-8").count("\n")}
                for path in sorted(directory.iterdir()) if path.is_file()}

    def overview(self) -> list[dict]:
        """One row per measured build for corpus-level charts."""
        return [{
            "id": p["id"], "wave": p.get("wave") or "wave1", "split": p["split"],
            "count": p["input_features"]["functionality_count"],
            "level": p["input_features"].get("composition_level", "BT"),
            "industry": p["input_features"].get("industry"),
            "total_tokens": p["build_token_usage"]["total_tokens"],
            "calls": p["build_token_usage"]["provider_call_count"],
        } for p in self.measured]
