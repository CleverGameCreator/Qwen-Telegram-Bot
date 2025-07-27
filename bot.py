import asyncio
import logging
import os
import json
import aiohttp
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv(override=True)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
logging.info(f"Raw TELEGRAM_BOT_TOKEN from env (repr): {repr(TELEGRAM_BOT_TOKEN)}")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Railway specific variables
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "")  # Railway will provide this
WEBHOOK_PATH = f"/webhook/{TELEGRAM_BOT_TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}" if WEBHOOK_HOST else ""

# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# User preferences storage (in-memory, resets on restart)
user_prefs = {}

WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "")  # Railway will provide this
WEBHOOK_PATH = f"/webhook/{TELEGRAM_BOT_TOKEN}"
# Добавляем поддержку явного указания полного URL вебхука
WEBHOOK_URL = os.getenv("WEBHOOK_URL", f"{WEBHOOK_HOST}{WEBHOOK_PATH}" if WEBHOOK_HOST else "")

async def on_startup(bot: Bot):
    """Setup webhook on startup."""
    if WEBHOOK_URL:
        await bot.set_webhook(WEBHOOK_URL)
        logging.info(f"Webhook set to: {WEBHOOK_URL}")

async def invoke_llm_api(user_content: str, show_thoughts: bool) -> str:
    """Calls the OpenRouter API and returns the streamed response."""
    if not OPENROUTER_API_KEY:
        return "Ошибка: Токен OPENROUTER_API_KEY не найден в .env"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("REFERER_URL", "http://localhost"),
        "X-Title": os.getenv("TITLE_NAME", "Qwen3 Telegram Bot")
    }
    body = {
        "model": "qwen/qwen3-235b-a22b:free",
        "messages": [
            {
                "role": "system",
                "content": """Отвечай коротко и лаконично, как это принято в чатах, используй эмоджи где это уместно.\n                Строго не используй разметку Markdown!"""
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.7
    }
    full_response = ""
    api_url = "https://openrouter.ai/api/v1/chat/completions" # Убрал лишний пробел в конце URL
    try:
        # Используем async with для автоматического закрытия сессии
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logging.error(f"API request failed with status {response.status}: {error_text}")
                    return f"Ошибка при обращении к API: {response.status}"
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data)
                            if chunk_data.get("choices") and chunk_data["choices"][0].get("delta"):
                                content = chunk_data["choices"][0]["delta"].get("content")
                                if content:
                                    full_response += content
                        except json.JSONDecodeError:
                            logging.error(f"Error decoding JSON chunk: {data}")
                        except Exception as e:
                            logging.error(f"Error processing chunk: {e}")
    except aiohttp.ClientError as e:
        logging.error(f"API request failed: {e}")
        return f"Ошибка при обращении к API: {e}"
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return "Произошла непредвиденная ошибка."
    # Filter out <think> tags if user preference is set to False
    if not show_thoughts:
        full_response = re.sub(r'<think>.*?</think>\s*', '', full_response, flags=re.DOTALL | re.IGNORECASE).strip()
    return full_response if full_response else "Не удалось получить ответ от модели."

@dp.message(CommandStart())
async def send_welcome(message: types.Message):
    """Handles the /start command."""
    await message.reply("Привет! Отправь мне сообщение, и я постараюсь ответить с помощью новейшей модели Qwen3 235B.")

@dp.message(Command("think"))
async def toggle_think(message: types.Message):
    """Toggles the display of thought process (if available)."""
    user_id = message.from_user.id
    current_pref = user_prefs.get(user_id, {"show_thoughts": False}) # Default to False
    new_pref = not current_pref["show_thoughts"]
    user_prefs[user_id] = {"show_thoughts": new_pref}

    status = "включено" if new_pref else "выключено"
    await message.reply(f"Отображение размышлений {status}.")

@dp.message()
async def handle_message(message: types.Message):
    """Handles incoming text messages and replies using the LLM API."""
    if not message.text:
        return

    user_id = message.from_user.id
    show_thoughts = user_prefs.get(user_id, {}).get("show_thoughts", False)

    # Indicate that the bot is processing
    processing_message = await message.reply("Обрабатываю ваш запрос...")

    response_text = await invoke_llm_api(message.text, show_thoughts) # Changed function call

    # Delete the processing message
    await bot.delete_message(chat_id=processing_message.chat.id, message_id=processing_message.message_id)

    # Send the final response
    if response_text:
        # Filter out <think> tags if user preference is set to False
        if not show_thoughts:
            # Use regex to remove <think>...</think> blocks, including newlines inside
            response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()

        if not response_text: # Check if response is empty after filtering
             await message.reply("Ответ содержал только размышления, которые скрыты.")
             return

        # Split long messages if necessary (Telegram limit is 4096 chars)
        for i in range(0, len(response_text), 4096):
            await message.reply(response_text[i:i+4096])
    else:
        await message.reply("Не удалось получить ответ.")

async def main():
    """Starts the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logging.error("Telegram bot token not found. Set TELEGRAM_BOT_TOKEN in .env file.")
        return
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())