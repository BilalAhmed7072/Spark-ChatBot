import os
from groq import Groq
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY:", GROQ_API_KEY)

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print("❌ Error initializing Groq client:", e)
    client = None

def generate_llm_response(prompt: str, context: str = "", model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a professional, concise, and natural response for Spark Solutionz support.
    Automatically switches between RAG (context) and conversational mode.
    """
    if not client:
        return "LLM client is not initialized. Please check your GROQ_API_KEY."

    if context and context.strip():
        full_prompt = f"""
        You are Spark AI Assistant for Spark Solutionz a global software and AI company.
        Use the context below to answer the user's query accurately and professionally. Keep the answer under 4 sentences unless necessary. And do not use greetings pharases in response to every query unless it is necessary.


Context:
{context}

User Query:
{prompt}

If the context doesn't contain the answer, respond briefly based on your general knowledge,
but stay relevant to Spark Solutionz services and technologies.
"""
    else:
        full_prompt = f"""
You are Spark AI Assistant, representing Spark Solutionz.
The user said: "{prompt}"

Reply in a natural, polite, human-like tone, not robotic or overly cheerful. Respond conversationally but briefly under 4 sentences.
If it's a greeting or casual message, respond conversationally. Also Avoid repeating greetings or phrases like "nice to chat with you.
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Spark Solutionz AI Assistant. Be concise , polite and relevant"},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.6,
            max_tokens=300
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        print("❌ Error in LLM call:", e)
        return "Sorry, I'm having trouble generating a response right now."
