import os
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from config_loader import load_config

config = load_config()
DEFAULT_MODEL = config["default_llm_model"]
DEFAULT_TEMPERATURE = config["temperature"]

SYSTEM_PROMPT_TEMPLATE = """You are "{app_name}", a highly secure, privacy-first, local-only AI assistant.

Your task is to answer the user's question using ONLY the provided document context.

Strict Guidelines:
1. Answer the question using ONLY the facts explicitly stated in the retrieved context below. Do not extrapolate, assume, or use any pre-trained external knowledge.
2. If the context is empty, does not contain the answer, or only partially answers the question, you MUST reply exactly with this phrase:
"Information not found locally."
Do not include any explanation, warning, or additional text. Only output that exact phrase.
3. Keep your answers direct, concise, and factual.
4. Do not refer to the context or retrieved documents directly in your response (e.g., do not write "According to the context..."). Just state the facts.
"""

class LocalLLMClient:
    def __init__(self, model_name: str = None, temperature: float = None):
        self.model_name = model_name if model_name is not None else DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.llm = None
        self.initialize_client()

    def initialize_client(self):
        """Initializes the Ollama client with current settings."""
        # Set context size to 8192 to allow large context RAG retrieval
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_ctx=8192,
        )

    def change_model(self, new_model_name: str):
        """Changes the active model and re-initializes client."""
        if self.model_name != new_model_name:
            self.model_name = new_model_name
            self.initialize_client()

    def query(self, prompt: str, context_text: str) -> str:
        """Sends the system instructions, context, and user prompt to the local LLM."""
        if not self.llm:
            self.initialize_client()

        config_data = load_config()
        app_name = config_data.get("app_name", "Digital Twin")
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(app_name=app_name)

        formatted_user_prompt = f"Retrieved Context:\n{context_text}\n\nUser Question: {prompt}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=formatted_user_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Error communicating with local LLM server: {e}"
