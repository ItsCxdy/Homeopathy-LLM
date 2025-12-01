import os
import requests
import json
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import logging
# Using the langchain_community package imports as they were in the original file
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
DESCRIBING_SYMPTOMS = 1

class TelegramHomeopathyBot:
    def __init__(self):
        # Initialize embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = Chroma(
            persist_directory="./vector_db",
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Store user sessions: includes chat_history
        self.user_sessions = {}
    
    def get_user_session(self, user_id):
        """Get or create user session, initializing history."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'chat_history': [],
                'consultation_count': 0,
                'last_query': None  # Track query context
            }
        return self.user_sessions[user_id]
    
    def get_relevant_context(self, query, history):
        """
        Retrieve relevant context from the vector database using the latest query 
        and the accumulated user history for better RAG results.
        """
        # Create a composite query string from previous user inputs and the latest query
        user_history = [msg['content'] for msg in history if msg['role'] == 'user']
        history_summary = " ".join(user_history)
        
        # Combine the user's history and current message for a comprehensive retrieval query
        composite_query = f"Patient's case summary: {history_summary} {query}" if history_summary else query
        
        docs = self.retriever.invoke(composite_query) 
        context = "\n".join([doc.page_content for doc in docs])
        return context
        
    def query_ai(self, user_message, context, history):
        """Query the AI model via OpenRouter"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        # FIX: Add immediate check for API key presence inside the query function
        if not api_key:
            logger.error("🚨 CRITICAL API ERROR: OPENROUTER_API_KEY is missing or empty during function call.")
            return "❌ Configuration Error: The AI service API key (OPENROUTER_API_KEY) is missing. Please check your setup."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com", # For OpenRouter logging
            "X-Title": "Homeopathy Telegram Bot" # For OpenRouter logging
        }
        
        # System prompt reflecting the strict constraints from doctor_bot.py
        system_prompt = """You are an expert Homeopathy Doctor. Your goal is to find the best possible remedy from the provided Context.

**DIAGNOSTIC PROCESS & CONSTRAINTS:**
1. **Focus:** Analyze the patient's full symptom set, including the **Chat History**. Only analyze the symptoms explicitly mentioned by the patient. IGNORE any symptoms found *only* in the Context that the patient has not mentioned.
2. **Clarification:** You MUST conclude the diagnosis and suggest a remedy within the first **two or three turns** of the conversation. Ask a MAXIMUM of 3 concise clarifying questions in the first turn only, if needed. In subsequent turns, prioritize diagnosing based on the accumulated history.
3. **Prescription Rule:** **MUST** prescribe the single best-matching remedy when:
   a) You have clear matching symptoms from the Context.
   b) The patient explicitly asks for the medicine, or indicates they cannot answer more questions. In this case, use the best available information from the Chat History to prescribe.
4. **Safety:** If unsure or the context doesn't contain a relevant remedy, admit it honestly.
5. **Tone:** Always be professional, caring, and responsible.
"""
        # Build the message payload including the system instruction and the history
        messages = [
            {"role": "system", "content": system_prompt},
            *history, # Insert existing conversation history
            {"role": "user", "content": f"Context from homeopathy book:\n{context}\n\nPatient Complaint: {user_message}"}
        ]
        
        data = {
            "model": "meta-llama/llama-3-8b-instruct", 
            "messages": messages,
            "temperature": 0.2, 
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "No detailed error message.")
                
                # --- FIX: Added specific 401 check for better console logging ---
                if response.status_code == 401:
                    logger.error("🚨 CRITICAL API ERROR (401 Unauthorized) 🚨: The OPENROUTER_API_KEY is likely invalid or missing. Please check your .env file.")
                # ------------------------------------------------------------------
                
                logger.error(f"OpenRouter Error {response.status_code}: {error_message}")
                return f"❌ I'm having technical difficulties. Please try again later. (Error: {response.status_code})"
                
        except Exception as e:
            logger.error(f"Connection Error: {e}")
            return f"❌ Connection error. Please try again. Error: {str(e)}"

# Create bot instance
homeopathy_bot = TelegramHomeopathyBot()

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message when the command /start is issued."""
    user = update.effective_user
    welcome_text = f"""
👋 Hello *{user.first_name}*! I'm your AI Homeopathy Doctor 🤖

I can help you analyze symptoms and suggest potential homeopathic remedies based on medical knowledge.

💡 *How to use:*
1. Describe your symptoms in detail
2. I'll ask clarifying questions (max 3 in the first turn)
3. I'll suggest potential homeopathic remedies within 3 turns

⚠️ *Important Disclaimer:*
This is for educational purposes only. Always consult a qualified homeopath or medical professional for proper diagnosis and treatment.

Type your symptoms below to begin...
    """
    
    # Reset session for a clean start
    user_id = update.effective_user.id
    homeopathy_bot.user_sessions[user_id] = {'chat_history': [], 'consultation_count': 0}
    
    await update.message.reply_text(welcome_text, parse_mode='Markdown')
    return DESCRIBING_SYMPTOMS

async def handle_symptoms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user's symptom description and continue the diagnosis."""
    user_id = update.effective_user.id
    user_input = update.message.text
    
    # Get user session
    session = homeopathy_bot.get_user_session(user_id)
    session['consultation_count'] += 1
    
    # Send typing action
    await update.message.reply_chat_action(action="typing")
    
    # Send processing message
    processing_msg = await update.message.reply_text("🔍 Analyzing your symptoms...")
    
    try:
        # Get relevant context (using the history to enrich the retrieval query)
        context_text = homeopathy_bot.get_relevant_context(user_input, session['chat_history'])
        
        # Get AI response
        response = homeopathy_bot.query_ai(user_input, context_text, session['chat_history'])
        
        # Update chat history
        session['chat_history'].append({"role": "user", "content": user_input})
        session['chat_history'].append({"role": "assistant", "content": response})
        
        # Keep only last 6 messages (3 user + 3 assistant) to prevent context overflow
        if len(session['chat_history']) > 6:
            session['chat_history'] = session['chat_history'][-6:]
        
        # Send response using Markdown for clear formatting
        try:
            await processing_msg.delete()  # Remove processing message
        except Exception as e:
            logger.warning(f"Could not delete processing message: {e}")
        await update.message.reply_text(
            f"🩺 *Homeopathy Doctor:*\n\n{response}",
            parse_mode='Markdown' # Use Markdown for formatting
        )
        
        # Add quick actions
        quick_actions = [["🔍 Describe more symptoms or answer questions"], ["🔄 Start a new consultation"]]
        reply_markup = ReplyKeyboardMarkup(quick_actions, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("What would you like to do next?", reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await processing_msg.delete()
        await update.message.reply_text("❌ Sorry, I encountered an error. Please try again.")
    
    return DESCRIBING_SYMPTOMS

async def handle_quick_actions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle quick action buttons."""
    user_input = update.message.text
    
    if user_input in ["🔍 Describe more symptoms or answer questions", "Reset please"]:
        await update.message.reply_text("Please provide any additional details or clarify the Doctor's previous questions...")
    elif user_input in ["🔄 Start a new consultation", "Reset please"]:
        user_id = update.effective_user.id
        if user_id in homeopathy_bot.user_sessions:
            # Clear history and reset embeddings for a completely fresh start
            homeopathy_bot.user_sessions[user_id] = {
                'chat_history': [],
                'consultation_count': 0,
                'last_query': None  # Reset any cached query context
            }
            # Reinitialize the retriever to clear any cached context
            homeopathy_bot.retriever = homeopathy_bot.vector_store.as_retriever(search_kwargs={"k": 4})
        await update.message.reply_text("🔄 Starting new consultation. Please describe your symptoms...")
    
    return DESCRIBING_SYMPTOMS

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the conversation."""
    user_id = update.effective_user.id
    if user_id in homeopathy_bot.user_sessions:
        del homeopathy_bot.user_sessions[user_id]
        
    await update.message.reply_text(
        "👋 Consultation ended. Thank you for using Homeopathy AI Doctor!\n\n"
        "Remember to consult a qualified homeopath for proper treatment. 🌿",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message."""
    help_text = """
🤖 *Homeopathy AI Doctor Bot Help*

*Available Commands:*
/start - Begin a new consultation (resets history)
/help - Show this help message
/cancel - End current consultation

*How to get the best results:*
• Describe symptoms in detail
• Mention location, intensity, and timing
• Share what makes symptoms better/worse
• Be specific about associated feelings

*Disclaimer:* This bot provides educational information only. Always consult qualified medical professionals.
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

# FIX: Add a new handler to catch unhandled text messages and guide the user.
async def generic_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles any plain text message received outside the ConversationHandler."""
    await update.message.reply_text(
        "👋 Welcome! To begin a new consultation and describe your symptoms, please use the */start* command. Thank you!",
        parse_mode='Markdown'
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors and send a friendly message."""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.message:
        await update.message.reply_text(
            "❌ Sorry, I encountered an unexpected error. Please try again or use /start to begin a new consultation."
        )

def check_dependencies():
    """Checks for required environment variables and the vector database."""
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        print("❌ TELEGRAM_BOT_TOKEN not found in .env file. Please set it to run the bot.")
        return False
    
    if not os.path.exists("./vector_db"):
        print("❌ Vector database not found. Please run 'python ingest_book.py' first.")
        return False
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found in .env file. Please set it to run the bot.")
        return False
    
    return True

def main():
    """Start the bot."""
    
    if not check_dependencies():
        return
    
    print("🚀 Starting Telegram Homeopathy Bot...")
    
    # Get Telegram bot token from environment variable
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    # Create Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            DESCRIBING_SYMPTOMS: [
                MessageHandler(filters.Regex(r'^(🔍|🔄)'), handle_quick_actions),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_symptoms),
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    
    # Add handlers
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('cancel', cancel))
    
    # FIX: This handler catches any plain text not handled by the ConversationHandler or other commands.
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generic_message))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Start the Bot
    print("✅ Bot is running... Press Ctrl+C to stop")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    # Ensure all initialization warnings are visible before starting the main loop
    main()