from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from retriever import Retriever
from generator import Generator
import os
from dotenv import load_dotenv

load_dotenv()

class TelegramBot:
    def __init__(self, token):
        self.retriever = Retriever()
        self.generator = Generator()
        print("Закончили инициализацию Ретривера и Генератора")
        self.app = Application.builder().token(token).build()
        
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Привет! Я медицинский чат-бот. Задавайте вопросы на русском языке.")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_input = update.message.text

        retrieved_contexts = self.retriever.get_retrieved_answer(user_input)

        response = self.generator.generate_response(user_input, retrieved_contexts)
        await update.message.reply_text(response)
    
    def run(self):
        self.app.run_polling()

if __name__ == "__main__":
    TOKEN = os.getenv("TG")
    bot = TelegramBot(TOKEN)
    bot.run()