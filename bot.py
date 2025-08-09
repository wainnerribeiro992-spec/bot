import os
import discord
import requests
from io import BytesIO

# Variáveis de ambiente
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def gerar_imagem(prompt):
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        print(f"Erro Hugging Face: {response.status_code} - {response.text}")
        return None

@client.event
async def on_ready():
    print(f"Bot conectado como {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!img "):
        prompt = message.content[5:]
        await message.channel.send(f"🎨 Gerando imagem para: **{prompt}**...")

        image_data = gerar_imagem(prompt)

        if image_data:
            await message.channel.send(file=discord.File(image_data, "imagem.png"))
        else:
            await message.channel.send("❌ Não consegui gerar a imagem. Tente novamente.")

client.run(DISCORD_TOKEN)
