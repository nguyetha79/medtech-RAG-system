import sys
import os

import textwrap
from langchain_core.prompts import ChatPromptTemplate
import ollama

sys.path.append(os.path.dirname(__file__))
from retrieve import RetrievedChunk

class LLMGenerator:
    def __init__(self, model_name, max_new_tokens: int,
                 temperature: float, rag_prompt_template: str):
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.rag_prompt_template = rag_prompt_template

    def build_prompt(self, query: str, contexts: list[RetrievedChunk], history: list, verbose: bool = False,) -> str:
        """Builds the prompt for the LLM using the query and retrieved contexts."""
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

        if verbose:
            print("=" * 60)
            print("RETRIEVED CHUNKS")
            print("=" * 60)
            for chunk in contexts:
                print(f"  Rank {chunk.rank} | score={chunk.score:.3f}")
                print(textwrap.fill(chunk.text, width=70,
                                    initial_indent="    ",
                                    subsequent_indent="    "))
                print()
            print("=" * 60)
            print("PROMPT SENT TO LLM")
            print("=" * 60)
            print(prompt)
            print("=" * 60)
        return prompt

    def generate(self, prompt: str) -> str:
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
    
