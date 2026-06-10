from typing import Dict, List, Any, Optional
import torch

class MockVectorDatabase:
    """Mock ChromaDB/Faiss interface for semantic memory lookup."""
    def __init__(self):
        self.data = {
            "obstacle_identification": [
                {"rules": ["avoid_obstacle_radius_2.0", "slow_down_on_approach"]},
                {"rules": ["climb_above_if_static"]}
            ],
            "navigation": [
                {"rules": ["prefer_direct_path", "maintain_altitude_5m"]}
            ]
        }

    def similarity_search(self, query_type: str, features: torch.Tensor, top_k: int = 3) -> List[Dict[str, Any]]:
        # Returns mock data based on query_type
        return self.data.get(query_type, [])[:top_k]

class SemanticMemoryInterface:
    """RAG lookup and injection triggered by ambiguity."""
    def __init__(self):
        self.db = MockVectorDatabase()

    def query(self, query_type: str, visual_features: torch.Tensor, top_k: int = 3) -> Dict[str, Any]:
        results = self.db.similarity_search(query_type, visual_features, top_k)

        # Aggregate rules
        all_rules = []
        for res in results:
            all_rules.extend(res.get("rules", []))

        return {"rules": all_rules}

class ConstraintPropagator:
    """Injects rules as constraints into the language lobe."""
    def __init__(self):
        self.active_constraints = []

    def propagate(self, context: Dict[str, Any], priority: float = 0.5):
        rules = context.get("rules", [])
        for rule in rules:
            self.active_constraints.append({
                "rule": rule,
                "priority": priority
            })
        return self.active_constraints
