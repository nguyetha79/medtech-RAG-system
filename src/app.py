from rag_pipeline import RAGPipeline
import gradio as gr

"""
RAG User Interface implementation

Retrieved from:
https://github.com/ed-donner/llm_engineering/blob/main/week5/app.py

Author: Edward Donner
Licensed under MIT
"""

rag_pipeline = RAGPipeline()

def format_context(context_chunks) -> str:
    """Format retrieved context chunks into HTML for display

    Args:
        context_chunks: List of context chunk objects with rank, score, and text

    Returns:
        str: Formatted HTML string displaying the context chunks
    """
    for chunk in enumerate(context_chunks):
        result += f"<span style='color: #ff7800;'>Chunk {chunk.rank} (Score: {chunk.score:.2f})</span>\n\n"
        result += chunk.text + "\n\n"
    return result


def chat(history) -> tuple[list, str]:
    """Process a chat message and generate a response using the RAG pipeline

    Args:
        history: List of chat history dictionaries with 'role' and 'content'

    Returns:
        tuple: Updated history list and formatted context string
    """

    if isinstance(last_user_message, dict):
        last_user_message = str(last_user_message.get("text", ""))
    elif isinstance(last_user_message, list):
        last_user_message = str(last_user_message[0])
    else:
        last_user_message = str(last_user_message)

    # Convert Gradio history (list of dicts) to RAG pipeline history (list of tuples)
    rag_chat_history = []
    for i in range(0, len(history) - 1, 2): 
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            rag_chat_history.append((history[i]["content"], history[i+1]["content"]))

    answer, contexts = rag_pipeline.ask(last_user_message, rag_chat_history)

    # Append the assistant's response to the Gradio history format
    history.append({"role": "assistant", "content": answer})

    formatted_context = format_context(contexts)

    return history, formatted_context


def main():
    def put_message_in_chatbot(message, history) -> tuple[str, list]:
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="MedTech's Expert Assistant") as ui:
        gr.Markdown("# 🏢 MedTech's Expert Assistant\nAsk me anything about Troubleshooting!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                    buttons=["copy"],
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Troubleshooting...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme, share=True)


if __name__ == "__main__":
    main()