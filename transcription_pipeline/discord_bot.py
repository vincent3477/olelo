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


async def summarize_meeting(transcripts: str, participant_list: str = None):
    class TeamMemberUpdates(BaseModel):
        speaker_id: str = Field(description = "ID of the speaker responsible for task/ project.")
        project_name: str = Field(description = "The project they are assigned to (if stated)")
        accomplishments: str = Field(description = "What they have accomplished")
        to_do: str = Field(description="What are they planning to do next")
        blockers: str = Field(description="their list of blockers, if any")

    class MemberList(BaseModel):
        list_of_persons: list[TeamMemberUpdates] = Field(description = "A list of all topics that were discussed")


    class ProjectName(BaseModel):
        project_name: str = Field("The name of the project") # this should be cross referenced from previous meeting notes
        project_updates: str = Field("What are the updates regarding the project.")


    class ProjectList(BaseModel):
        project_list: list[ProjectName] = Field("List of all projects")



    class NameAttributor(BaseModel):
        person_name: str = Field( description="The real name of the person. leave as 'speaker_{int}' if it cannot be attributed.")
        speaker_id: str = Field(pattern=r"^speaker_\d{2}$", description="The speaker ID as provided in the transcript.")
        reason_for_matching: str = Field(format, description="A concrete explanation of why you matched speaker_ID with the name of the person")

    class NameList(BaseModel):
        name_list: list[NameAttributor] = Field("List of all attributed names")

    name_attribute_instructions = """
        You are a speaker-name attribution agent.
    
        Your task is to match speaker IDs in a meeting transcript to the correct
        participant names using evidence found in the transcript.
    
        You will be given:
    
        1. A comma-separated list of possible meeting participants.
        2. A transcript containing speaker IDs and their utterances.
    
        Your goal is to return a dictionary mapping each speaker ID to a person name.
    
        IMPORTANT PRINCIPLE:
        Only assign a name to a speaker when there is sufficient evidence in the
        transcript to support the attribution.
    
        Do NOT guess. If the evidence is weak, ambiguous, indirect, or contradictory,
        leave the speaker's name empty.
    
        ====================
        VALID EVIDENCE
        ====================
    
        Strong evidence includes:
    
        1. Self-identification
        Example:
        Speaker_01: "Hi everyone, my name is Vincent."
    
        Result:
        Speaker_01 -> "Vincent"
    
        2. Direct identification by another speaker
        Example:
        Speaker_02: "Vincent, what do you think?"
        Speaker_01: "I think we should continue."
    
        This may support Speaker_01 -> "Vincent" ONLY if the conversational
        context clearly indicates that Speaker_01 is responding to being addressed.
    
        3. Explicit introduction
        Example:
        Speaker_03: "This is AJ, and I'll be presenting today."
    
        Result:
        Speaker_03 -> "AJ"
    
        4. A speaker being explicitly called upon and immediately responding
        Example:
        Speaker_01: "AJ, can you give us your update?"
        Speaker_04: "Sure. I finished the task yesterday."
    
        This is strong contextual evidence that Speaker_04 may be "AJ",
        provided there is no ambiguity.
    
        ====================
        INSUFFICIENT EVIDENCE
        ====================
    
        Do NOT assign a name based only on:
    
        - Writing style
        - Topic knowledge
        - Personality
        - Assumptions about gender
        - Frequency of speaking
        - The order of names in the participant list
        - A name merely appearing somewhere in the transcript
        - Weak conversational guesses
        - A speaker responding when it is unclear who was being addressed
    
        If multiple people could reasonably match a speaker, leave the speaker empty.
    
        ====================
        CONSTRAINTS
        ====================
    
        - Each speaker ID can be assigned to AT MOST one person.
        - Each person can be assigned to AT MOST one speaker ID.
        - Only use names from the provided participant list.
        - Do not invent names.
        - If a speaker cannot be identified with sufficient confidence, speaker id for that speaker.
        - Evidence in the transcript takes priority over assumptions.
        - If evidence conflicts, do not assign the name unless the conflict can be
        clearly resolved.
        """
    attribute_extraction_agent = Agent(name="Name attribute extractor", instructions=name_attribute_instructions, output_type=NameList)
    person_name_output = Runner.run_sync(attribute_extraction_agent, f"extract here {raw_transcript}").final_output

    print(person_name_output)

    attributed_transcript = raw_transcript

    try:
        for item in person_name_output.name_list:
            print(type(item.speaker_id))
            if item.speaker_id != "":
                raw_transcript = attributed_transcript.lower().replace(item.speaker_id, item.person_name)
    except:
        validated_container = NameList.model_validate_json(person_name_output)
        for item in validated_container.name_list:
            print(item.person_name, item.speaker_id)



    member_extractor_instructions = """You are a meeting segmentation agent. Your job is to split the entire meeting into sections based on the speaker discussing their updates. 
    You need to include the following details:
    - the ID of the speaker (this field must be in the format Speaker_{id_here})
    - The name of the project or projects they are assigned to
    - What they have accomplished in which project
    - What are their next plans in which project
    - What blockers do they have (if any)
    Please do not attribute names that may be present in the transcript to the speaker id. 

    For each 
    Return a json format
    """
    agent = Agent(name = "Meeting Segmentation Agent", instructions=member_extractor_instructions, output_type=MemberList)
    raw_transcript = create_transcription_person_summary(transcripts=transcripts, list_participants=participant_list)
    person_ind_updates = Runner.run_sync(starting_agent = agent, input = f"segment this meeting {attributed_transcript}").final_output
    print(person_ind_updates)







    

    #print(person_name_output)


    project_extractor_instructions = """You are the project extractor agent. Your respnsibility is to extract all projects that were mentioned in the above mentioned speaker-specific summary"""
    project_extraction_agent = Agent(name="Project extraction agent", instructions=project_extractor_instructions, output_type=ProjectList)
    project_updates = Runner.run_sync(project_extraction_agent, f"extract here {person_ind_updates}").final_output

    return project_updates, person_ind_updates

    



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



    


async def send_transcripts(message):
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
    #download_token = body["download_token"]
    #headers = {"Authorization": f"Bearer {download_token}"}
    #recording_files = body["payload"]["object"]['recording_files']


    global processor

    if processor == None:
        raise Exception("Processor not initialized")
    print(body)

    s = processor.print_something()

    

    s = """# # Meeting Summary

**Topic:** Overview of Earth's Climate Systems and Recent Trends
**Date:** August 6, 2026
**Duration:** 75 minutes

## Attendees

* Dr. Laura Chen
* Prof. Michael Ortiz
* Sarah Kim
* David Brooks
* Nina Patel

## Agenda

* Review of Earth's major climate zones
* Discussion of factors influencing climate
* Recent observations and long-term trends
* Public outreach and educational initiatives

## Discussion

The meeting began with a review of Earth's primary climate zones, including tropical, arid, temperate, continental, polar, and highland climates. Participants discussed how latitude, elevation, ocean currents, prevailing winds, and topography interact to produce distinct regional weather patterns.

The group examined the role of major ocean circulation systems in regulating global temperatures. It was noted that warm and cold ocean currents influence coastal climates, while large-scale atmospheric circulation helps distribute heat and moisture around the planet.

Attendees also discussed recent climate observations, including changes in average temperatures, precipitation patterns, and the frequency of certain extreme weather events in various regions. The group emphasized the importance of interpreting long-term climate records alongside short-term weather variability when communicating scientific findings.

The conversation shifted to climate monitoring technologies, including satellite observations, weather stations, ocean buoys, and climate models. Participants highlighted the value of combining multiple data sources to improve understanding of global climate processes and to support research and forecasting efforts.

Finally, the team discussed opportunities to improve public education by creating accessible materials that explain the difference between weather and climate, illustrate how Earth's climate system functions, and encourage informed discussions based on scientific evidence.

## Decisions

* Develop updated educational resources explaining Earth's climate systems.
* Expand collaboration between researchers and educators on climate communication.
* Continue monitoring long-term climate indicators using multiple observational datasets.

## Action Items

* **Dr. Chen:** Prepare a summary of recent climate observations for public outreach.
* **Prof. Ortiz:** Review educational materials for scientific accuracy.
* **Sarah:** Gather visual examples of major global climate zones.
* **David:** Compile recent satellite and ocean monitoring data for the next meeting.
* **Nina:** Draft a proposal for a climate education workshop aimed at high school students.

## Next Meeting

The team will reconvene next month to review progress on educational materials, discuss new observational data, and evaluate opportunities for collaboration with local schools and community organizations.
"""

    await store_meeting_summary(random.randint(1, 255), "Scrum_Meeting", s)
    await send_transcripts(s)


    
    
    """
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

                    


                    fs = await asyncio.to_thread(post_process_pipeline.post_process(record_filename)) 

                    #await store_meeting_sumamry()


                    await send_transcripts(fs)

                    
                    
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
    """
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
