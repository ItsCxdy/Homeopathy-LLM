import os
import requests
import json
from dotenv import load_dotenv
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Reverting to community package imports as in the original code
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

class HomeopathyDoctorBot:
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
        
        # Initialize chat history for state management
        self.chat_history = []
        
    def get_relevant_context(self, query):
        """Retrieve relevant context from the vector database"""
        # Using the modern .invoke() method remains correct for the retriever object
        docs = self.retriever.invoke(query) 
        context = "\n".join([doc.page_content for doc in docs])
        return context
        
    def query_ai(self, user_message, context, history):
        """Query the AI model via Chute.ai"""
        api_key = os.getenv("CHUTEAI_API_KEY")
        
        if not api_key:
            logger.error("CHUTEAI_API_KEY is missing or empty in the environment.")
            return "Error: Configuration missing CHUTEAI_API_KEY."
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # System prompt remains the same
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
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": f"Context from homeopathy book:\n{context}\n\nPatient Complaint: {user_message}"}
        ]
        
        data = {
            # FIX: Replaced the likely incorrect model name ("Gemma 3 4b It") 
            # with a known working model from the user's Chute.ai example, 
            # as the 404 error is usually caused by an invalid model identifier.
            "model": "meituan-longcat/LongCat-Flash-Chat-FP8", 
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 500
        }
        
        # --- FIX: Corrected hostname from llm.chute.ai to llm.chutes.ai ---
        api_url = "https://llm.chutes.ai/v1/chat/completions"

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "No detailed error message.")
                logger.error(f"Chute.ai API Error Status: {response.status_code}. Message: {error_message}")
                return f"Error: Unable to get response (Status: {response.status_code}). Chute.ai Message: {error_message}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection/Request Error: {e}")
            return f"Error: Failed to connect to Chute.ai. Check your connection or API endpoint. Details: {str(e)}"
        
    def start_consultation(self):
        
        print("🤖 Homeopathy AI Doctor is ready!")
        print("=" * 50)
        print("Please describe your symptoms... (Type 'quit' to exit)")
        print()
        
        while True:
            try:
                user_input = input("Patient: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nHomeopathy Doctor: Thank you for consulting! Wishing you good health! 🌿")
                    break
                
                if not user_input:
                    print("Homeopathy Doctor: Please describe your symptoms.")
                    continue
                
                print("Homeopathy Doctor: Analyzing your symptoms...")
                
                # --- Get relevant context ---
                context = self.get_relevant_context(user_input)
                
                # --- Get AI response ---
                # Pass the history to the query function
                response = self.query_ai(user_input, context, self.chat_history)
                
                print(f"Homeopathy Doctor: {response}\n")
                print("-" * 50)

                # --- Update chat history with patient input and bot response ---
                self.chat_history.append({"role": "user", "content": user_input})
                # Check if the response contains an error message before appending as assistant's content
                if not response.startswith("Error:"):
                    self.chat_history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\n\nHomeopathy Doctor: Consultation ended. Take care!")
                break
            except Exception as e:
                print(f"Homeopathy Doctor: I encountered an issue. Please try again. Error: {str(e)}")

def main():
    print("🚀 Starting Homeopathy AI Doctor Setup...")
    
    # Check if vector database exists
    if not os.path.exists("./vector_db"):
        print("❌ Vector database not found.")
        print("💡 Please run 'python ingest_book.py' first to create the knowledge base.")
        return
    
    # Check API key
    if not os.getenv("CHUTEAI_API_KEY"):
        print("❌ CHUTEAI_API_KEY not found in .env file.")
        return
    
    try:
        bot = HomeopathyDoctorBot()
        bot.start_consultation()
    except Exception as e:
        print(f"❌ Failed to start bot: {str(e)}")

if __name__ == "__main__":
    main()