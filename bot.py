import os
import requests
import discord
import asyncio

# Variáveis de ambiente (preenchidas no Render)
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SPACE_URL = os.getenv(
    "SPACE_URL",
    "https://api-inference.huggingface.co/spaces/SG161222/Realistic_Vision_V5.1"
)

# Headers para autenticação na Hugging Face
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Configuração do Discord
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"🤖 Bot conectado como {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!imagem "):
        prompt = message.content[8:].strip()
        await message.channel.send(f"🎨 Gerando imagem para: *{prompt}*...")

        try:
            response = requests.post(
                SPACE_URL,
                headers=headers,
                json={"inputs": prompt},
                timeout=120  # até 2 minutos de espera
            )

            if response.status_code == 200 and response.content:
                with open("saida.png", "wb") as f:
                    f.write(response.content)
                await message.channel.send(file=discord.File("saida.png"))
            else:
                await message.channel.send(
                    f"❌ Erro ao gerar imagem: {response.status_code} - {response.text}"
                )

        except requests.exceptions.RequestException as e:
            await message.channel.send(f"⚠️ Erro de conexão com o Space: {e}")

async def start_bot():
    await client.start(DISCORD_TOKEN)

# Criar loop para o server.py usar
loop = asyncio.get_event_loop()
loop.create_task(start_bot())
