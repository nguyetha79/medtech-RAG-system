from rag_pipeline import RAGPipeline
import gradio as gr

rag_pipeline = RAGPipeline()

def format_context(context_chunks):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for i, chunk in enumerate(context_chunks):
        result += f"<span style='color: #ff7800;'>Chunk {chunk.rank} (Score: {chunk.score:.2f})</span>\n\n"
        result += chunk.text + "\n\n"
    return result

def chat(history):
    last_user_message = history[-1]["content"]

    if isinstance(last_user_message, dict):
        last_user_message = str(last_user_message.get("text", ""))
    elif isinstance(last_user_message, list):
        last_user_message = str(last_user_message[0])
    else:
        last_user_message = str(last_user_message)

    # Convert Gradio history (list of dicts) to RAG pipeline history (list of tuples)
    rag_chat_history = []
    for i in range(0, len(history) - 1, 2): # Iterate over prior user-assistant pairs
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            rag_chat_history.append((history[i]["content"], history[i+1]["content"]))

    answer, contexts = rag_pipeline.ask(last_user_message, rag_chat_history)

    # Append the assistant's response to the Gradio history format
    history.append({"role": "assistant", "content": answer})

    formatted_context = format_context(contexts)

    return history, formatted_context

def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="MedTechs Expert Assistant") as ui:
        gr.Markdown("# 🏢 MedTechs Expert Assistant\nAsk me anything about Troubleshooting!")

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