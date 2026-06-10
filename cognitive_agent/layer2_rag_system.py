import torch
import os
from typing import Dict, List, Any, Optional
from .utils import log_info, log_error

# Real DB imports
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False

# Mock ROS 2 if not available
try:
    from rclpy.node import Node
    from std_srvs.srv import Trigger # Placeholder for semantic lookup service
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    class Node:
        def __init__(self, name): self.name = name
        def create_service(self, *args, **kwargs): return None

class VectorDatabaseInterface:
    """Production interface for ChromaDB."""
    def __init__(self, db_path: str = "./chroma_db"):
        if not HAS_RAG_DEPS:
            log_error("ChromaDB or SentenceTransformers not found. Falling back to mock.")
            self.mock_mode = True
            return

        self.mock_mode = False
        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.client.get_or_create_collection(name="cognitive_memory")

    def add_document(self, text: str, metadata: Dict[str, Any], doc_id: str):
        if self.mock_mode: return
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def similarity_search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.mock_mode:
            return [{"rules": ["avoid_obstacle_radius_2.0"]}]

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        # Flatten results for easy consumption
        formatted = []
        if results['metadatas']:
            for meta in results['metadatas'][0]:
                formatted.append(meta)
        return formatted

class SemanticMemoryInterface(Node):
    """RAG system integrated with ROS 2 Service."""
    def __init__(self):
        super().__init__('semantic_memory')
        self.db = VectorDatabaseInterface()

        if HAS_ROS2:
            # Service to allow Visual Lobe to trigger retrieval
            self.srv = self.create_service(Trigger, 'query_semantic_memory', self.query_callback)

    def query_callback(self, request, response):
        # Triggered by high entropy in Visual Lobe
        # response.message = self.query(...)
        response.success = True
        return response

    def query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        results = self.db.similarity_search(query_text, top_k)
        all_rules = []
        for res in results:
            if "rules" in res:
                all_rules.extend(res["rules"] if isinstance(res["rules"], list) else [res["rules"]])
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
