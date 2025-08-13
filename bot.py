import os
import torch
import discord
import asyncio
from huggingface_hub import login
from diffusers import StableDiffusionPipeline

# Códigos ANSI para cores no terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Tokens
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "SEU_TOKEN_DISCORD")
HF_TOKEN = os.getenv("HF_TOKEN", "SEU_TOKEN_HUGGINGFACE")

print(f"{YELLOW}🟢 Container iniciado! Preparando ambiente...{RESET}")

# Login na Hugging Face
print(f"{BLUE}🔑 Fazendo login na Hugging Face...{RESET}")
login(token=HF_TOKEN)

# Intents do Discord
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Carregando modelo
print(f"{YELLOW}🖼️  Carregando modelo de IA... Isso pode levar 1-3 minutos.{RESET}")
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-2")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

print(f"{GREEN}✅ Modelo carregado com sucesso!{RESET}")

@client.event
async def on_ready():
    print(f"{GREEN}🤖 Bot conectado como {client.user}{RESET}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!imagem "):
        prompt = message.content[8:].strip()

        await message.channel.send(f"🎨 Gerando imagem para: *{prompt}*...")
        image = pipe(prompt).images[0]
        image.save("saida.png")

        await message.channel.send(file=discord.File("saida.png"))

async def start_bot():
    await client.start(DISCORD_TOKEN)

loop = asyncio.get_event_loop()
loop.create_task(start_bot())
