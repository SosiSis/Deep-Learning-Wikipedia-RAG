import gradio as gr
from src.app_cli import ask

with gr.Blocks() as demo:
    gr.Markdown("# Deep Learning Wikipedia RAG ")
    q = gr.Textbox(label="Ask a question about deep learning")
    out = gr.Markdown()

    def _run(question):
        try:
            return ask(question)
        except Exception as e:
            return f"Error: {e}"

    q.submit(_run, q, out)

if __name__ == "__main__":
    demo.launch()
