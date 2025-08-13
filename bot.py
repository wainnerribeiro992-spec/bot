import os
import asyncio
import discord
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
from flask import Flask
import threading

# --- Servidor Web falso para Render ---
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot do Discord está rodando!"

def run_web():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# --- Tokens e configurações ---
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/stable-diffusion-2")  # Modelo padrão

if not DISCORD_TOKEN or not HF_TOKEN:
    raise ValueError("Erro: variáveis de ambiente DISCORD_TOKEN e HF_TOKEN não foram definidas.")

# Login Hugging Face
login(token=HF_TOKEN)

# Configurações do Discord
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

pipe = None

@client.event
async def on_ready():
    global pipe
    print(f"✅ Bot conectado como {client.user}")
    print(f"🔄 Carregando modelo de IA: {MODEL_NAME} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    print(f"✅ Modelo '{MODEL_NAME}' carregado com sucesso!")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!imagem "):
        prompt = message.content[8:].strip()

        await message.channel.send(f"🎨 Gerando imagem usando `{MODEL_NAME}` para: *{prompt}*...")
        image = pipe(prompt).images[0]
        image.save("saida.png")

        await message.channel.send(file=discord.File("saida.png"))

async def run_bot():
    while True:
        try:
            await client.start(DISCORD_TOKEN)
        except discord.errors.HTTPException as e:
            if e.status == 429:
                print("⚠ Rate limit detectado! Esperando 60 segundos...")
                await asyncio.sleep(60)
            else:
                print(f"⚠ Erro HTTPException: {e}")
                await asyncio.sleep(10)
        except Exception as e:
            print(f"⚠ Erro inesperado: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    threading.Thread(target=run_web).start()  # Inicia servidor web em paralelo
    asyncio.run(run_bot())
