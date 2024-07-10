import os

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def calculate_cost(text_string, model_id):
    cost_tier = {
        "tts-1": 0.015,
        "tts-1-hd": 0.03,
    }
    cost_unit = cost_tier.get(model_id, None)
    if cost_unit is None:
        return None
    return cost_unit * len(text_string) / 1000


client = OpenAI(api_key=OPENAI_API_KEY)

text_input = """
Esse é um teste de texto para ver se a API do OpenAI está funcionando corretamente. O texto é bem simples e não deve ter problemas.
"""

model = "tts-1"
voice = "echo"

response = client.audio.speech.create(model=model, voice=voice, input=text_input)

request_cost = calculate_cost(text_input, model)
print(f"Request cost: ${request_cost:.3f}")
response.write_to_file("./output/audio.mp3")
