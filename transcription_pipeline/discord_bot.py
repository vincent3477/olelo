"""
reference - https://builtin.com/software-engineering-perspectives/discord-bot-python
"""

import logging
import os
import numpy as np
import os
import sys
import discord
from discord.ext import commands
import base64
import hashlib
import hmac
import nltk
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
import spacy
import requests
from post_process import post_process_pipeline
from agents import Agent, Runner, function_tool
from pymongo import MongoClient
os.environ["KMP_DUPLICATE_OK"] = "TRUE"
# standard practice if working in multi-threaded environments
os.environ["OMP_NUM_THREADS"] = "1" 
import faiss
from sentence_transformers import SentenceTransformer
import random
from pydantic import Field, BaseModel
from llm_ext_prompts import create_transcription_person_summary


client = MongoClient("mongodb://localhost:27017/")
db = client["meeting_database"]
collection = db["meeting_db_collection"]

model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-v2')

dimensions = 768

some_agent = Agent(name = "assistant", instructions = "You are a helpful assistant")

nlp = spacy.load("en_core_web_sm")

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.messages = True
intents.guild_messages = True
intents.message_content = True
processor = None

import os

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET") 
CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")
CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")

RETRYABLE_ERROR_CODES = { 408, 429, 500, 502, 503, 504}
REDIRECT_CODES = {301, 302, 303, 307, 308}

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")



print("File loaded. __name__ =", __name__)

bot = commands.Bot(command_prefix="!", intents = intents)

app = FastAPI(title="sample")

logger = logging.getLogger("uvicorn.error")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # This ensures the traceback is still printed to your terminal console
    logger.error("Unhandled exception occurred", exc_info=exc)
    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})

async def embed_summary(id: str, meeting_name: str, summary_body:str):
    global dimensions

    filename = os.path.join("vector_db_store",meeting_name,"embeddings.bin")
    temp = os.path.join("vector_db_store",meeting_name,"embeddings.tmp")
    if os.path.exists(filename):
        index = faiss.read_index(filename)
        # grab the file
        embeddings = model.encode(summary_body, batch_size= 16)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis = 0)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        id_array = np.array(np.array([id]),dtype=np.int64)
        index.add_with_ids(embeddings, id_array)
    else:
        index = faiss.IndexIDMap(faiss.IndexFlatL2(768))
        # grab the file
        embeddings = model.encode(summary_body, batch_size= 16)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis = 0)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        print(embeddings)
        id_array = np.array(np.array([id]),dtype=np.int64)
        index.add_with_ids(embeddings, id_array)

    faiss.write_index(index, temp)
    os.replace(temp, filename)
    return


async def get_summary(meeting_name, query: str):
    query_embedding = model.encode(query)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    
    filename = os.path.join("vector_db_store",meeting_name,"embeddings.bin")
    print(filename)
    if os.path.exists(filename):
        index = faiss.read_index(filename)
        
        _, i = index.search(np.array(query_embedding), 1)
        return i
    return []


async def store_meeting_summary(id: str, meeting_name: str, summary_body:str):
    global collection
    try:
        collection.insert_one({"meeting_id":id, "meeting_name": meeting_name, "summary_body":summary_body})
        await embed_summary(id = id, meeting_name=meeting_name, summary_body=summary_body)
    except Exception as e:
        print(e, file=sys.stderr)


## Discord Bot∂
def split_text(summary):
    print(" at aplit")
    doc = nlp(summary)
    print(" at aplis")

    
    chunks = []
    chunks.append("----------------------------------------------------")


    max_len = 1800
    summ_str = ""

    for s in doc.sents:
        if len(s.text) + len(summ_str) <= max_len:
            summ_str += s.text
        else:
            chunks.append(summ_str)
            summ_str = s.text 
        
    chunks.append("----------------------------------------------------")
    return chunks

@bot.event
async def on_ready():
	# CREATES A COUNTER TO KEEP TRACK OF HOW MANY GUILDS / SERVERS THE BOT IS CONNECTED TO.
	guild_count = 0

	# LOOPS THROUGH ALL THE GUILD / SERVERS THAT THE BOT IS ASSOCIATED WITH.
	for guild in bot.guilds:
		# PRINT THE SERVER'S ID AND NAME.
		print(f"- {guild.id} (name: {guild.name})")

		# INCREMENTS THE GUILD COUNTER.
		guild_count = guild_count + 1

	# PRINTS HOW MANY GUILDS / SERVERS THE BOT IS IN.
	print("SampleDiscordBot is in " + str(guild_count) + " guilds.")



@bot.command()
async def ping(ctx, meeting_name: str, query: str):
    global collection
    try:
    # Set a strict 30-second timeout
        indexes = await get_summary(meeting_name = meeting_name, query=query)
        #result = await asyncio.wait_for(Runner.run(some_agent, arg), timeout=30.0)
        if len(indexes) == 0:
            await ctx.send("There is nothing n the db rn.")
        else:
            print(indexes)

            c = collection.find({})

            for i in indexes[0]:
                summary = collection.find_one({"meeting_id":int(i)})
            


            #s = result.final_output
                await send_transcripts(summary['summary_body'])
    except asyncio.TimeoutError:
        print("Agent call timed out after 30 seconds!")
        s = "Sorry, the assistant took too long to respond."
    except Exception as e:
        print(f"Agent error: {e}")
        s = f"An error occurred: {e}"
    print("finish")



    

async def send_transcripts(message: str | list[str]):
	# CHECKS IF THE MESSAGE THAT WAS SENT IS EQUAL TO "HELLO".

    channel_id = 1506427721608073248
     
    channel = bot.get_channel(channel_id)
    if isinstance(message, str): 
        chunks = split_text(message)
        await bot.wait_until_ready()
        for i in chunks:
            print("sending", i)
            await channel.send(i)

    elif isinstance(message, list[str]): 
        await bot.wait_until_ready()
        for i in message:
            print("sending", i)
            await channel.send(i)


async def send_transcripts_chunked(message):
	# CHECKS IF THE MESSAGE THAT WAS SENT IS EQUAL TO "HELLO".

    channel_id = 1506427721608073248
     
    channel = bot.get_channel(channel_id)
    
    chunks = split_text(message)
    await bot.wait_until_ready()
    for i in chunks:
        print("sending", i)
        await channel.send(i)
    

   

async def run_discord_bot():
    print('STARTING DISCORD BOT')
    await bot.start(DISCORD_TOKEN)
		
async def run_api():
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        reload=True,
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_bot():
    print("we are now running the bot")
    await bot.start(DISCORD_TOKEN)


## Zoom webhook
def get_access_token():
    ## GET AN ACCESS TOKEN
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {"Authorization": f"Basic {encoded_credentials}"}

    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={ACCOUNT_ID}"

    response = requests.post(url, headers=headers)
    response.raise_for_status()

    return response.json()["access_token"]



async def get_audio_file_summarize(body):
    download_token = body["download_token"]
    headers = {"Authorization": f"Bearer {download_token}"}
    recording_files = body["payload"]["object"]['recording_files']


    global processor

    if processor == None:
        raise Exception("Processor not initialized")
    print(body)

    s = processor.print_something()


    # need to change random rand int, as it this can cause hash collisions
    #await store_meeting_summary(random.randint(1, 255), "Scrum_Meeting", s)
    #await send_transcripts(s)


    
    
    
    # Download each recording file
    for f in recording_files:
        record_name = f["id"]
        extension = f["file_extension"].lower()
        location = f["download_url"]
        if extension == "m4a" or extension == "mp3":
            for attempt in range(5):
                try:  
                    download_req_response = await asyncio.to_thread(requests.get, location, headers=headers, stream=True, allow_redirects=False)
                    
                    if download_req_response in RETRYABLE_ERROR_CODES and download_req_response not in REDIRECT_CODES:
                        raise RuntimeError(f"Error {download_req_response.status_code} while trying to request download")


                    if download_req_response.status_code in REDIRECT_CODES:
                        location = download_req_response.headers.get("Location")
                        if not location:
                            raise RuntimeError(f"No location header: {download_req_response.status_code}")
                        continue


                    download_req_response.raise_for_status()

                    record_filename = f"{record_name}.{extension}"

                    with open(record_filename, "wb") as file_out:
                        for chunk in download_req_response.iter_content(chunk_size=8192):
                            if chunk:
                                file_out.write(chunk)

                    


                    final_proj_lists, final_pers_lists = await asyncio.to_thread(post_process_pipeline.post_process_openai(audio_file=record_filename)) 

                    #await store_meeting_sumamry()

                    await send_transcripts(final_pers_lists)
                    await send_transcripts(final_proj_lists)

                    
                    
                    break # Finish execution when everything went successfully.
                
                except RuntimeError as e:
                    print(e)
                    await asyncio.sleep(5)
                    if attempt == 4:
                        print("This is the last retry. Quitting.")
                        break

                except Exception as e:
                    print(e)
                    return 1
    
    return 0

@app.post("/webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):

    body = await request.json()
    print(body)
    if body.get("event") == "endpoint.url_validation": ## WEBHOOK VALIDATION

        key = ZOOM_WEBHOOK_SECRET.encode()

        plain_token = body["payload"]["plainToken"]
        message = plain_token.encode()

        encrypted_token = hmac.new(key, message, hashlib.sha256).hexdigest()
        # hash digest is the fixed-size binary output from a crypto hash function (but this time we are doing a hex string format)

        return {"plainToken":plain_token, "encryptedToken":encrypted_token}
    
    if body.get("event") == "meeting.ended": ## MEETING RECORDING RECEIVED
        print("recording was done")
        background_tasks.add_task(get_audio_file_summarize, body)

        return {"ok": True}



async def main():

    global processor
    processor = await asyncio.to_thread(post_process_pipeline)
    
    await asyncio.gather(
        run_bot(),
        run_api(),
        
    )


if __name__ == "discord_bot":
    asyncio.run(main())
