import os
import torch
from prompts import generate_chunk_prompt, generate_final_prompt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoFeatureExtractor
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydub import AudioSegment
from tempfile import TemporaryDirectory
from transformers import AutoProcessor, AutoModelForImageTextToText
from agents import Agent, Runner
from openai import OpenAI
from pydantic import BaseModel, Field, TypeAdapter
from llm_ext_prompts import create_transcription_person_summary, get_member_attribute_prompt

client = OpenAI()

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

class post_process_pipeline():
    def __init__(self):

        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Pyannote
        self.pyannote_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=os.getenv("HF_API_KEY"))
        self.pyannote_pipeline.to(self.device)

        # Whisper API
        """torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3-turbo"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
            return_timestamps=True
        )
"""
        # gemma
        #self.gemma_processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it", token = os.getenv('HF_API_KEY'))
        #self.gemma_model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-12b-it", token = os.getenv('HF_API_KEY'))
        #self.gemma_model.to(self.device)


    def get_whisper_model(self):        
        return self.whisper_pipeline
        
    def query_gemma(self, query):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ]
            },
        ]
        inputs = self.gemma_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.gemma_model.device)

        outputs = self.gemma_model.generate(**inputs, max_new_tokens=8192)
        summary = self.gemma_processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        return summary


    def diar_transcribe(self, audio_file: str):
        with TemporaryDirectory() as tmp:
            wav_aud_file = os.path.join(tmp, "wav_audio_file.wav")
            audio = AudioSegment.from_file(audio_file, format="m4a")
            audio.export(wav_aud_file, format="wav") 
            with ProgressHook() as hook:
                diarizations = self.pyannote_pipeline(wav_aud_file, hook = hook)
            transcriptions = self.whisper_pipeline(wav_aud_file)
        return diarizations, transcriptions

    def calc_overlap(self, d_time_start, d_time_end, t_time_start, t_time_end):
        overlap = max(0.0, min(d_time_end, t_time_end) - max(d_time_start, t_time_start))
        duration = max(t_time_end - t_time_start, 1e-8)
        return overlap / duration

    
    def merge_trans_diar(self, diarizations, transcriptions):
        for t in transcriptions["chunks"]:
            t_time_start = t["timestamp"][0]
            t_time_end = t["timestamp"][1]
            best_overlap_score = 0.0
            for s, speaker in diarizations.speaker_diarization:
                d_time_start = s.start
                d_time_end = s.end
                curr_score = self.calc_overlap(d_time_start, d_time_end, t_time_start, t_time_end)
                if curr_score > best_overlap_score:
                    t["speaker"] = speaker
                    best_overlap_score = curr_score
        unmerged = transcriptions["chunks"]
        merged = []
        last = len(unmerged)
        i = 0
        j = 1
        while(i < last and j < last):
            if unmerged[i].get("speaker") == unmerged[j].get("speaker") and i != j:
                j += 1
            elif j - i > 1:
                start_time = unmerged[i]["timestamp"][0]
                speaker_str = ""
                speaker = unmerged[i]["speaker"]
                while (i < j - 1):
                    # merge everything starting at j until i
                    speaker_str += unmerged[i]["text"] + " "
                    i += 1
                end_time = unmerged[i]["timestamp"][0]
                turn = {'timestamp': (start_time, end_time), 'text': speaker_str, 'speaker': speaker}
                merged.append(turn)
            else:
                # this is when all the speakers between i and j are different
                merged.append(unmerged[i])
                j += 1
                i += 1
        merged.append(unmerged[last - 1])
        return merged


    def chunk_transcript(self, transcript, est_length = 180):
        chunks = []
        chunk = []
        b = 0
        start = transcript[b]["timestamp"][0]
        while(b < len(transcript)):
          if transcript[b]["timestamp"][1] - start > 180:
            chunk.append(transcript[b])
            chunks.append(chunk)
            for i in chunk:
              print(i)
            chunk = []
            start = transcript[b]["timestamp"][0]
          else:
            chunk.append(transcript[b])
            b += 1
        chunks.append(chunk)
        return chunks


    def summarize_chunk(self,transcript):
        structure = generate_chunk_prompt(transcript=transcript)
        summary = self.query_gemma(structure)
        print(summary)
        return summary


    def generate_final_summary(self, chunks):
        structure = generate_final_prompt(chunks=chunks)
        summary = self.query_gemma(structure)
        print(summary)
        return summary


    def post_process_gemma(self, audio_file):
        d, t = self.diar_transcribe(audio_file)
        merged = self.merge_trans_diar(d, t)
        chunk_transcripts = self.chunk_transcript(merged)
        chunk_summaries = {}
        index = 0
        for i in chunk_transcripts:
            chunk_summaries[index]=(self.summarize_chunk(i))
            index += 1
        fs = self.generate_final_summary(chunk_summaries)
        print(fs)
        return fs

    def post_process_openai(self, audio_file, list_participants = None) -> list[str]:

        # creates a transcript, then uses openai api to summarize, then chunk meetings first by individual updates then sectioned into projects. 

        d, t = self.diar_transcribe(audio_file)
        merged = self.merge_trans_diar(d, t)

        raw_transcript = create_transcription_person_summary(transcripts=merged, list_participants=list_participants)

        #name_attribute_instructions = get_member_attribute_prompt()
        #attribute_extraction_agent = Agent(name="Name attribute extractor", instructions=name_attribute_instructions, output_type=NameList)
        #person_name_output = Runner.run_sync(attribute_extraction_agent, f"extract here {raw_transcript}").final_output

        person_name_output = client.beta.chat.completions.parse(
            model = "gpt-5-mini",
            messages = [
                {"role": "system", "content": get_member_attribute_prompt()},
                {"role": "user", "content": f"You are given a list of participants and a raw transcript. Match names and speaker_id according to the systems instructions: {raw_transcript}"}
            ],
            format = NameList
        )
    
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

        individual_updates = client.beta.chat.completions.parse(
            model = "gpt-5-mini",
            messages = [
                {"role": "system", "content": member_extractor_instructions},
                {"role": "user", "content": f"Given the instructions above, extract, by speaker, the name of the project(s), what they have accomplished, next plans, and blockers: {raw_transcript}"}
            ],
            format = MemberList
        )

        #agent = Agent(name = "Meeting Segmentation Agent", instructions=member_extractor_instructions, output_type=MemberList)
        
        #person_ind_updates = Runner.run_sync(starting_agent = agent, input = f"segment this meeting {attributed_transcript}").final_output
        print(individual_updates)


      

        project_extractor_instructions = """You are the project extractor agent. Your respnsibility is to extract all projects that were mentioned in the above mentioned speaker-specific summary"""
        #project_extraction_agent = Agent(name="Project extraction agent", instructions=project_extractor_instructions, output_type=ProjectList)
        #project_updates = Runner.run_sync(project_extraction_agent, f"extract here {individual_updates}").final_output

        project_updates = client.beta.chat.completions.parse(
            model = "gpt-5-mini",
            messages = [
                {"role": "system", "content": project_extractor_instructions},
                {"role": "user", "content": f"Given the instructions above, extract all projects that were given in the individual updates summary: {individual_updates}"}
            ],
            format = ProjectList
        )

        final_proj_lists = []
        final_string = ""


        for item in project_updates.project_list:
            if len(item.project_name) + len(item.project_updates) + len(final_string) < 1950:
                final_string += item.project_name + "\n"
                final_string += item.project_updates + "\n\n"
            else:
                final_proj_lists.append(final_string)
                final_string = ""

        if len(final_string) > 0:
            final_proj_lists.append(final_string)

        final_pers_lists = []
        final_string = ""

        for item in individual_updates.list_of_persons:
            if len(item.speaker_id) + len(item.project_name) + len(item.accomplishments) + len(item.to_do) + len(item.blockers) +  len(final_string) < 1950:
                final_string += item.speaker_id + "\n"
                final_string += item.project_name + "\n"
                final_string += item.accomplishments + "\n"
                final_string += item.to_do + "\n"
                final_string += item.blockers + "\n\n"
            else:
                final_pers_lists.append(final_string)
                final_string = ""
        if len(final_string) > 0:
            final_pers_lists.append(final_string)


        

            


        return  final_proj_lists, final_pers_lists
    


    def print_something(self):
        return "this was returned."