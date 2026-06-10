import torch
import os
import shutil
from cognitive_agent.layer2_rag_system import VectorDatabaseInterface

def test_chroma_db_integration():
    db_path = "./test_chroma"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = VectorDatabaseInterface(db_path=db_path)
    if not db.mock_mode:
        db.add_document("Test Rule", {"rules": "test_action"}, "id_1")
        results = db.similarity_search("Test", top_k=1)
        assert len(results) > 0
        assert results[0]["rules"] == "test_action"

    if os.path.exists(db_path):
        shutil.rmtree(db_path)
