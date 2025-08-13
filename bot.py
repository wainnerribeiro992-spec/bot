import os
from huggingface_hub import login
import torch
from diffusers import StableDiffusionPipeline
import discord
import asyncio
from flask import Flask

# Tokens de autenticação
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# Modelo padrão (pode ser alterado via comando)
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-2")

# Login na Hugging Face
login(token=HF_TOKEN)

# Discord intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

print(f"Carregando modelo: {MODEL_ID}...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

@client.event
async def on_ready():
    print(f"Bot conectado como {client.user}")

@client.event
async def on_message(message):
    global pipe, MODEL_ID

    if message.author == client.user:
        return

    # Gerar imagem
    if message.content.startswith("!imagem "):
        prompt = message.content[8:].strip()
        await message.channel.send(f"Gerando imagem para: *{prompt}*...")
        image = pipe(prompt).images[0]
        image.save("saida.png")
        await message.channel.send(file=discord.File("saida.png"))

    # Trocar modelo
    if message.content.startswith("!modelo "):
        novo_modelo = message.content[8:].strip()
        try:
            await message.channel.send(f"Carregando modelo: {novo_modelo}...")
            pipe = StableDiffusionPipeline.from_pretrained(
                novo_modelo,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            MODEL_ID = novo_modelo
            await message.channel.send(f"Modelo alterado para: **{novo_modelo}**")
        except Exception as e:
            await message.channel.send(f"Erro ao carregar modelo: {e}")

# Flask server para manter o bot vivo no Render
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot está rodando!"

async def start_bot():
    await client.start(DISCORD_TOKEN)

loop = asyncio.get_event_loop()
loop.create_task(start_bot())
app.run(host="0.0.0.0", port=10000)
