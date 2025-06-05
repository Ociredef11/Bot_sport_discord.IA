import discord
from discord.ext import commands
from model import get_class
import os, random
import requests

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def sport(ctx):
    '''Riconosce lo sport nell'immagine e lo descrive'''
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            await attachment.save(f"./{attachment.filename}")
            risposta = get_class(f"./{attachment.filename}")
            await ctx.send(risposta)
    else:
        await ctx.send("⚠️ Devi allegare un'immagine per usare questo comando.")


@bot.command()
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_url = attachment.url
            await attachment.save(f"./{attachment.filename}")
            await ctx.send(get_class(model_path="./keras_model.h5", labels_path="labels.txt", image_path=f"./{attachment.filename}"))
    else:
        await ctx.send("You forgot to upload the image :(")


bot.run("MY TOKEN")
