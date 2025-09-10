import time

from openai import OpenAI

from sglang.utils import print_highlight

port = 30000
client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")

elapsed_times = []

for it in range(6):
    start_time = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this video?",
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "/home/geon/86CxyhFV9MI.mp4"
                        },
                    },
                ],
            }
        ],
        max_tokens=1 if it >= 3 else 300,
    )
    elapsed_time = time.time() - start_time
    elapsed_times.append(elapsed_time)
    print(f"Time: {elapsed_time}")

    print_highlight(response.choices[0].message.content)

elapsed_times = elapsed_times[3:]  # exclude warmup

print(f"Average time: {sum(elapsed_times) / len(elapsed_times)}")
