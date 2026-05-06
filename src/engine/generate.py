import sys
import os

from langchain_core.prompts import ChatPromptTemplate
import ollama

sys.path.append(os.path.dirname(__file__))

from retrieve import RetrievedChunk

class LLMGenerator:
    """Wraps an LLM model for prompt construction and generation"""

    def __init__(self, model_name: str, max_new_tokens: int,
                 temperature: float, rag_prompt_template: str):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.rag_prompt_template = rag_prompt_template

    def build_prompt(self, query: str, contexts: list[RetrievedChunk], history: list) -> str:
        """Build the prompt for the LLM using retrieved contexts and history

        Args:
            query (str): The user query
            contexts (list[RetrievedChunk]): Retrieved document chunks
            history (list): Conversation history as human/AI pairs

        Returns:
            str: The formatted prompt string
        """
        context_parts: list[str] = [
            f"[Document {chunk.rank}] {chunk.text}" for chunk in contexts
        ]
        context: str = "\n\n".join(context_parts)

        chat_history_str = ""
        if history:
            for human_message, ai_message in history:
                chat_history_str += f"Human: {human_message}\nAI: {ai_message}\n"

        prompt_template = ChatPromptTemplate.from_template(self.rag_prompt_template)
        prompt: str = prompt_template.format(
            chat_history=chat_history_str,
            context=context,
            question=query,
        )

        return prompt

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM for the provided prompt

        Args:
            prompt (str): The prompt to send to the LLM

        Returns:
            str: The generated response text
        """
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": self.temperature,
            }
        )

        return response["message"]["content"].strip()
    
