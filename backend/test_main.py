import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    resp = client.chat.completions.create(
        model="llama2-7b",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(resp.choices[0].message.content)
except Exception as e:
    print("LLM test failed:", e)
