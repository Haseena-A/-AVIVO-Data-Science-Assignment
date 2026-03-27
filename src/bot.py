"""
bot.py — Telegram bot with RAG + Vision capabilities
Usage: python bot.py
"""

import os
import io
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode, ChatAction

# Local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import get_db, init_db, load_knowledge_base, answer_query
from vision_engine import describe_image, format_vision_response
from history import HistoryManager

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "rag_store.db")
KB_DIR = os.getenv("KB_DIR", str(Path(__file__).parent.parent / "knowledge_base"))

_db_conn = None
_history: HistoryManager = None


def get_conn():
    global _db_conn
    if _db_conn is None:
        _db_conn = get_db(DB_PATH)
        init_db(_db_conn)
    return _db_conn


def get_history() -> HistoryManager:
    global _history
    if _history is None:
        _history = HistoryManager(get_conn())
    return _history


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

HELP_TEXT = """
🤖 *GenAI Assistant Bot*

I can answer questions from my knowledge base and describe images!

*Commands:*
`/ask <your question>` — Ask anything from the knowledge base
`/image` — Send this command then upload an image (or just send a photo)
`/history` — View your last 3 interactions
`/clear` — Clear your conversation history
`/summarize` — Summarize the last response
`/help` — Show this help message

*Tips:*
• For RAG, I search across Tech FAQs, Company Policies, Recipes, and AI concepts
• For images, I generate a caption + 3 tags
• Results are cached for speed ⚡
"""


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"👋 Hello {update.effective_user.first_name}!\n\n" + HELP_TEXT,
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args).strip()
    user_id = update.effective_user.id

    if not query:
        await update.message.reply_text(
            "❓ Usage: `/ask <your question>`\nExample: `/ask What is RAG?`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    history = get_history()
    history.add(user_id, "user", f"/ask {query}")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: answer_query(query, get_conn())
        )

        answer = result["answer"]
        sources = result["sources"]
        cached = result["cached"]

        # Format response
        source_line = ""
        if sources:
            source_line = "\n\n📚 *Sources:* " + ", ".join(f"`{s}`" for s in sources)

        cache_line = " _(cached ⚡)_" if cached else ""
        response = f"💬 *Answer:*{cache_line}\n\n{answer}{source_line}"

        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        history.add(user_id, "assistant", answer)

    except Exception as e:
        logger.exception(f"Error in /ask handler: {e}")
        await update.message.reply_text(f"❌ An error occurred: {str(e)}")


async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Send me an image now and I'll describe it!\n"
        "_Just upload a photo directly in this chat._",
        parse_mode=ParseMode.MARKDOWN,
    )
    # Set user state to expect image
    context.user_data["expecting_image"] = True


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos — either from /image flow or direct send."""
    user_id = update.effective_user.id
    history = get_history()

    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)

    try:
        # Get highest resolution photo
        photo = update.message.photo[-1]
        file = await photo.get_file()
        image_bytes = await file.download_as_bytearray()
        image_bytes = bytes(image_bytes)

        await update.message.chat.send_action(ChatAction.TYPING)

        history.add(user_id, "user", "[uploaded image for description]")

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: describe_image(image_bytes, "image/jpeg")
        )

        response = format_vision_response(result)
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        history.add(user_id, "assistant", response)

    except Exception as e:
        logger.exception(f"Error in photo handler: {e}")
        await update.message.reply_text(f"❌ Failed to process image: {str(e)}")

    finally:
        context.user_data.pop("expecting_image", None)


async def handle_document_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle images sent as documents (uncompressed)."""
    doc = update.message.document
    if not doc or not doc.mime_type.startswith("image/"):
        return

    user_id = update.effective_user.id
    history = get_history()
    mime_type = doc.mime_type

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        file = await doc.get_file()
        image_bytes = bytes(await file.download_as_bytearray())

        history.add(user_id, "user", f"[uploaded image document: {doc.file_name}]")

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: describe_image(image_bytes, mime_type)
        )

        response = format_vision_response(result)
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        history.add(user_id, "assistant", response)

    except Exception as e:
        logger.exception(f"Error in document image handler: {e}")
        await update.message.reply_text(f"❌ Failed to process image: {str(e)}")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = get_history()
    formatted = history.format_for_display(user_id)
    await update.message.reply_text(
        f"📜 *Your last interactions:*\n\n{formatted}",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    get_history().clear(user_id)
    await update.message.reply_text("🗑️ History cleared!")


async def cmd_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    last = get_history().get_last_context(user_id)

    if not last:
        await update.message.reply_text("No previous response to summarize.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": f"Summarize this in 2 sentences:\n\n{last}"
                }],
            )
            summary = msg.content[0].text.strip()
        except Exception as e:
            summary = last[:300] + ("..." if len(last) > 300 else "")
    else:
        summary = last[:300] + ("..." if len(last) > 300 else "")

    await update.message.reply_text(
        f"📋 *Summary:*\n\n{summary}",
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text messages — treat as /ask queries."""
    text = update.message.text.strip()
    if text.startswith("/"):
        return  # Ignore unknown commands

    # Simulate /ask
    context.args = text.split()
    await cmd_ask(update, context)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

async def post_init(application: Application):
    """Called after application starts — set bot commands menu."""
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Show help"),
        BotCommand("ask", "Ask a question from the knowledge base"),
        BotCommand("image", "Send an image for description"),
        BotCommand("history", "View your last interactions"),
        BotCommand("clear", "Clear your conversation history"),
        BotCommand("summarize", "Summarize the last response"),
    ]
    await application.bot.set_my_commands(commands)

    # Load knowledge base on startup
    logger.info("Loading knowledge base...")
    conn = get_conn()
    try:
        load_knowledge_base(KB_DIR, conn)
        logger.info("✅ Knowledge base ready.")
    except Exception as e:
        logger.error(f"KB load error: {e}")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required!")

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("summarize", cmd_summarize))

    # Photo and image-document handlers
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(
        MessageHandler(filters.Document.MimeType("image/png"), handle_document_image)
    )
    app.add_handler(
        MessageHandler(filters.Document.MimeType("image/jpeg"), handle_document_image)
    )
    app.add_handler(
        MessageHandler(filters.Document.MimeType("image/webp"), handle_document_image)
    )

    # Fallback: plain text → treat as question
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("🚀 Bot is starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
