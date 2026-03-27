"""
gradio_ui.py — Local debug UI for testing RAG + Vision without Telegram.

Usage:
    pip install gradio
    python gradio_ui.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys
import io
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine import get_db, init_db, load_knowledge_base, answer_query, retrieve_top_k
from vision_engine import describe_image, format_vision_response
from history import HistoryManager

import gradio as gr

KB_DIR = os.getenv("KB_DIR", str(Path(__file__).parent / "knowledge_base"))
DB_PATH = os.getenv("DB_PATH", "debug_ui.db")

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
conn = get_db(DB_PATH)
init_db(conn)
print("Loading knowledge base...")
load_knowledge_base(KB_DIR, conn)
print("✅ Knowledge base loaded.")

history = HistoryManager(conn)
DEBUG_USER_ID = 0  # Single user for local UI


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def rag_query(question: str, show_chunks: bool):
    if not question.strip():
        return "⚠️ Please enter a question.", ""

    history.add(DEBUG_USER_ID, "user", question)
    result = answer_query(question, conn)

    answer = result["answer"]
    sources = result["sources"]
    cached = result["cached"]

    tag = " *(cached ⚡)*" if cached else ""
    display = f"**Answer{tag}**\n\n{answer}\n\n---\n📚 Sources: {', '.join(sources) or 'none'}"
    history.add(DEBUG_USER_ID, "assistant", answer)

    chunks_display = ""
    if show_chunks:
        chunks = retrieve_top_k(question, conn, top_k=3)
        lines = []
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"**[{i}] {c['source']}** (score: {c['score']:.4f})\n\n{c['text']}\n"
            )
        chunks_display = "\n---\n".join(lines)

    return display, chunks_display


def vision_query(image):
    if image is None:
        return "⚠️ Please upload an image."

    from PIL import Image
    buf = io.BytesIO()
    if not isinstance(image, Image.Image):
        return "⚠️ Invalid image format."
    image.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    history.add(DEBUG_USER_ID, "user", "[image upload]")
    result = describe_image(image_bytes, "image/jpeg")
    response = format_vision_response(result)

    history.add(DEBUG_USER_ID, "assistant", response)
    return response


def get_history():
    return history.format_for_display(DEBUG_USER_ID)


def clear_history():
    history.clear(DEBUG_USER_ID)
    return "🗑️ History cleared."


def get_summarize():
    last = history.get_last_context(DEBUG_USER_ID)
    if not last:
        return "No previous response to summarize."

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=150,
                messages=[{"role": "user", "content": f"Summarize in 2 sentences:\n\n{last}"}],
            )
            return f"**Summary:**\n\n{msg.content[0].text.strip()}"
        except Exception as e:
            return f"LLM error: {e}\n\nFallback: {last[:300]}..."
    return f"**Last response (truncated):**\n\n{last[:400]}"


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="GenAI Bot Debug UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 GenAI Bot — Local Debug UI")
    gr.Markdown(
        "Test the **RAG** and **Vision** pipelines locally before connecting to Telegram. "
        "All interactions are stored in session history."
    )

    with gr.Tabs():

        # ---- RAG Tab ----
        with gr.Tab("📚 RAG — Ask"):
            with gr.Row():
                with gr.Column(scale=3):
                    q_input = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. What is cosine similarity?",
                        lines=2,
                    )
                    show_chunks = gr.Checkbox(label="Show raw retrieved chunks", value=False)
                    ask_btn = gr.Button("Ask", variant="primary")

                with gr.Column(scale=4):
                    answer_out = gr.Markdown(label="Answer")

            chunks_out = gr.Markdown(label="Retrieved chunks (debug)")

            ask_btn.click(
                rag_query,
                inputs=[q_input, show_chunks],
                outputs=[answer_out, chunks_out],
            )
            q_input.submit(
                rag_query,
                inputs=[q_input, show_chunks],
                outputs=[answer_out, chunks_out],
            )

        # ---- Vision Tab ----
        with gr.Tab("🖼️ Vision — Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload image", type="pil")
                    img_btn = gr.Button("Describe", variant="primary")
                with gr.Column(scale=2):
                    vision_out = gr.Markdown(label="Caption & Tags")

            img_btn.click(vision_query, inputs=[img_input], outputs=[vision_out])

        # ---- History Tab ----
        with gr.Tab("📜 Session History"):
            hist_out = gr.Markdown()
            with gr.Row():
                hist_btn = gr.Button("Refresh History")
                clear_btn = gr.Button("Clear History", variant="stop")
                summ_btn = gr.Button("Summarize Last Response")

            hist_btn.click(get_history, outputs=[hist_out])
            clear_btn.click(clear_history, outputs=[hist_out])
            summ_btn.click(get_summarize, outputs=[hist_out])

    gr.Markdown(
        "---\n"
        "**Models**: Embeddings via `all-MiniLM-L6-v2` · "
        "LLM via Claude Haiku or Ollama · "
        "Vision via Claude Vision or BLIP"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
