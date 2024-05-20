import discord
from discord.ext import commands
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification
from peft import AutoPeftModelForCausalLM, PeftModelForSequenceClassification, PeftConfig, PeftModel
import torch
import json


with open('config.json', 'r') as file:
    # Load the JSON data from the file into a Python object
    data = json.load(file)

print(data["TOKEN"])

TOKEN = data["TOKEN"]
MOD_DIR = data["MOD_DIR"]

# define label maps
id2label = {2: "Neither", 1: "Offensive", 0: "Hate Speech"}
label2id = {"Neither": 2, "Offensive": 1, "Hate Speech": 0}
# how to load peft model from hub for inference
config = PeftConfig.from_pretrained(MOD_DIR)
inference_model = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path, num_labels=3, id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(inference_model, MOD_DIR)




intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='!',intents=intents)

@client.event
async def on_ready():
    print(f'{client .user} has connected to Discord!')

@client.command()
async def hello(ctx, user: discord.User):
    await ctx.send(f'Hello {user.name}!')

# Event for when a message is sent in the server


# Press the green button in the gutter to run the script.




def check_Offensive(text_list):
    # define list of example
    id2label = {2: "Neither", 1: "Offensive", 0: "Hate Speech"}

    print("Untrained model predictions:")
    print("----------------------------")
    for text in text_list:
        # tokenize text
        inputs = tokenizer.encode(text, return_tensors="pt")
        # compute logits
        logits = model(inputs).logits
        # convert logits to label
        predictions = torch.argmax(logits)
        print(text + " - " + id2label[predictions.tolist()])
    return id2label[predictions.tolist()]


@client.event
async def on_message(message):
    res = check_Offensive([message.content])
    # This line is necessary if you also want to use commands with the bot
    if not(res == "Neither"):
        await message.channel.send(f"This Messaage is classified as {res} , {message.author} please refrain from using inappropriate language on the server")
    await client.process_commands(message)


# Ignore messages sent by the bot itself

client.run(TOKEN)






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
