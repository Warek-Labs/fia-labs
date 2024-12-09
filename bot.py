import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext, MessageHandler
import telegram.ext.filters as filters
from dotenv import load_dotenv

load_dotenv()

from responses import *


class TelegramBot:
    def __init__(self):
        key = os.environ.get('TELEGRAM_KEY')
        self.app = ApplicationBuilder().token(key).build()

    def start(self):
        self.app.add_handler(MessageHandler(filters.CHAT, self._handle_message))
        self.app.run_polling()

    async def _handle_message(self, update: Update, context: CallbackContext) -> None:
        msg = update.message.text
        chat_id = update.effective_chat.id

        try:
            res = generate_response(msg)
            await context.bot.send_message(chat_id=chat_id, text=res)
        except Exception as e:
            print(e)
            await context.bot.send_message(chat_id=chat_id, text=f'Error generating response: {e}')
