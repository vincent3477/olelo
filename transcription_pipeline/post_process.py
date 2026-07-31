import os
import torch
from prompts import generate_chunk_prompt, generate_final_prompt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoFeatureExtractor
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydub import AudioSegment
from tempfile import TemporaryDirectory
from transformers import AutoProcessor, AutoModelForImageTextToText

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


    def post_process(self, audio_file):
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

    def print_something(self):
        return "this was returned."