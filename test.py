import os
from dotenv import load_dotenv
from groq import Groq

def main():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file with GROQ_API_KEY=your_key or set the env var."
        )

    client = Groq(api_key=api_key)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "WAP to generate a star with triangle",
                }
            ],
            model="llama-3.3-70b-versatile",  # ✅ updated model
            temperature=0.2,
            max_tokens=1024,
        )

        # ✅ Use dot notation
        print(chat_completion.choices[0].message.content)

    except Exception as exc:
        print(f"Request failed: {exc}")

if __name__ == "__main__":
    main()
