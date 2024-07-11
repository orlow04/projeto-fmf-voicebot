import os

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

text_input = """
Olá, boa tarde! No Hospital BIAmigos, temos a Doutora Ana Souza, especialista em Pediatria.
Infelizmente, na segunda-feira às 16 horas, a Doutora Ana Souza estará indisponível.
No entanto, ela tem disponibilidade na segunda-feira das 08:00 às 09:00, das 09:00 às 10:00, das 14:00 às 15:00.
Se precisar de ajuda para agendar uma consulta em um desses horários, por favor,
forneça seu nome, CPF e telefone para podermos ajudá-lo da melhor forma. Estamos à disposição para auxiliá-lo no Hospital BIAmigos.
"""
model = "tts-1"
voice = "shimmer"

response = client.audio.speech.create(model=model, voice=voice, input=text_input)

response.write_to_file("./output/audio.mp3")
