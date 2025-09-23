import time

from openai import OpenAI

port = 8090
client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")

start_time = time.time()
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Who are you?",
                },
            ],
        }
    ],
    max_tokens=300,
)
elapsed_time = time.time() - start_time

print(response.choices[0].message.content)
