from bot import loop

# Mantém o bot rodando
try:
    loop.run_forever()
except KeyboardInterrupt:
    pass
