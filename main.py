from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

chat_response = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[
        {"role": "system", "content": "Bạn là trợ lý AI"},
        {"role": "user", "content": input("Nhập câu hỏi: ")},
    ]
)

print("Chat response:", chat_response.choices[0].message.content)