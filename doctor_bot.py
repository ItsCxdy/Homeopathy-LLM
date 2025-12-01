import os
import requests
import json
from dotenv import load_dotenv

# FIX 1: Reverting the import back to the community package, 
# as requested, to avoid the 'langchain_huggingface' ModuleNotFoundError.
# NOTE: This will bring back a deprecation warning, but the code will run, 
# provided 'sentence-transformers' is installed.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Note: Keeping Chroma from community for minimal change, but be aware of its deprecation warning.
# If you want to eliminate the Chroma warning, you would install 'langchain-chroma' and change this import:
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
        
        # --- FIX 2: Initialize chat history for state management ---
        self.chat_history = []
        
    def get_relevant_context(self, query):
        """Retrieve relevant context from the vector database"""
        # Using the modern .invoke() method remains correct for the retriever object
        docs = self.retriever.invoke(query) 
        context = "\n".join([doc.page_content for doc in docs])
        return context
        
    def query_ai(self, user_message, context, history):
        """Query the AI model via OpenRouter"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # --- FIX 3: ENHANCED SYSTEM PROMPT for history and strict questioning ---
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
                # Provide more helpful API error details
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "No detailed error message.")
                return f"Error: Unable to get response (Status: {response.status_code}. OpenRouter Message: {error_message})"
                
        except Exception as e:
            return f"Error: {str(e)}"
        
    def start_consultation(self):
        # self.chat_history is now initialized in __init__
        
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

                # --- FIX 4: Update chat history with patient input and bot response ---
                # The LLM needs the history to be in the 'role' format
                self.chat_history.append({"role": "user", "content": user_input})
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
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found in .env file.")
        return
    
    try:
        bot = HomeopathyDoctorBot()
        bot.start_consultation()
    except Exception as e:
        print(f"❌ Failed to start bot: {str(e)}")

if __name__ == "__main__":
    main()