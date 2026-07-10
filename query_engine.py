from typing import List, Dict, Any, Tuple
from ingest import get_db, _db_lock
from llm_client import LocalLLMClient
from config_loader import load_config

class QueryEngine:
    def __init__(self, model_name: str = None):
        config = load_config()
        active_model = model_name if model_name is not None else config["default_llm_model"]
        self.db = get_db()
        self.llm_client = LocalLLMClient(model_name=active_model)

    def change_model(self, model_name: str):
        """Allows switching models dynamically from the UI."""
        self.llm_client.change_model(model_name)

    def query(self, user_query: str, top_k: int = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes semantic search, injects context to system prompt,
        calls the local LLM, and returns the response with source citations.
        """
        config = load_config()
        k_val = top_k if top_k is not None else config["top_k"]
        # 1. Retrieve most similar document chunks
        try:
            with _db_lock:
                docs = self.db.similarity_search(user_query, k=k_val)
        except Exception as e:
            return f"Error querying local vector store: {e}", []

        if not docs:
            return "Information not found locally.", []

        # 2. Assemble context content and build source citations list
        context_parts = []
        citations = []
        seen_chunks = set()

        for doc in docs:
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page")
            
            # De-duplicate chunks that are identical
            chunk_sig = (source, page, doc.page_content[:150])
            if chunk_sig in seen_chunks:
                continue
            seen_chunks.add(chunk_sig)
            
            page_info = f" (Page {page})" if page else ""
            context_parts.append(f"--- Document: {source}{page_info} ---\n{doc.page_content}")
            
            citations.append({
                "source": source,
                "page": page,
                "content": doc.page_content
            })

        context_text = "\n\n".join(context_parts)

        # 3. Call local LLM
        response_text = self.llm_client.query(user_query, context_text)

        # 4. Strict check for "Information not found locally"
        cleaned_response = response_text.strip()
        # If response indicates empty knowledge, wipe out citations to be clean
        if "information not found locally" in cleaned_response.lower():
            # Standardize exact response
            cleaned_response = "Information not found locally."
            citations = []

        return cleaned_response, citations
