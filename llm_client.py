from typing import Iterator, List, Tuple
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

CONDENSE_PROMPT = """Rewrite the user's latest question so it can be understood without the conversation, resolving references like "it", "they", "that year" or "what about ..." from the earlier turns. Keep names, companies, years and figures exactly as written. If the latest question already stands on its own, return it unchanged. Do not answer it. Output only the question."""

# How much of each earlier answer the rewrite sees: enough to resolve references,
# small enough to keep the extra LLM call quick.
HISTORY_ANSWER_CHARS = 500


class LocalLLMClient:
    def __init__(self, model_name: str = None, temperature: float = None):
        self.model_name = model_name if model_name is not None else DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.llm = None
        self.initialize_client()

    def initialize_client(self):
        """Initializes the Ollama client with current settings."""
        # Context window in tokens; QueryEngine sizes the retrieved context to fit it
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_ctx=load_config().get("num_ctx", 8192),
        )

    def change_model(self, new_model_name: str):
        """Changes the active model and re-initializes client."""
        if self.model_name != new_model_name:
            self.model_name = new_model_name
            self.initialize_client()

    def stream(self, prompt: str, context_text: str) -> Iterator[str]:
        """Streams the answer to prompt, grounded in context_text, as the model
        generates it. Raises if the Ollama server or model can't be reached, so
        callers can tell an error apart from an answer."""
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

        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def rewrite_followup(self, question: str, history: List[Tuple[str, str]]) -> str:
        """Turns a follow-up like "what about 2022?" into a standalone question
        using the earlier (role, content) turns, so retrieval and the answer
        prompt (which never see the chat) know what it refers to. Falls back to
        the original question if the model's output doesn't look like one."""
        if not self.llm:
            self.initialize_client()
        turns = "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {content[:HISTORY_ANSWER_CHARS]}"
            for role, content in history
        )
        messages = [
            SystemMessage(content=CONDENSE_PROMPT),
            HumanMessage(content=f"Conversation:\n{turns}\n\nLatest question: {question}"),
        ]
        rewritten = self.llm.invoke(messages).content.strip().strip('"').strip()
        if rewritten.lower().startswith("latest question:"):
            rewritten = rewritten.split(":", 1)[1].strip()
        # An empty or rambling reply means the model answered instead of rewriting
        if not rewritten or len(rewritten) > max(300, 4 * len(question)):
            return question
        return rewritten
