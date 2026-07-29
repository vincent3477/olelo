# ʻŌlelo

**ʻŌlelo** (Hawaiian: *language, speech, expression*) is a custom large language model developed by the AIEA Lab to support research, teaching, and experimentation in explainable and neurosymbolic AI.

The goal of ʻŌlelo is not just to generate a custom LLM for the AIEA lab  to **support structured reasoning, transparency, and responsible use of language models** in large-group, academic settings.

---

## 🌺 About the Name

In ʻŌlelo Hawaiʻi, **ʻōlelo** refers to language, speech, and meaningful expression.  
The name reflects our view of language models as systems that *express knowledge*, rather than opaque black boxes that merely predict tokens.

---

## 🎯 Project Goals

ʻŌlelo is designed to:
- Listen to meetings and produce meeting notes
- Be a custom tutor and helper for the lab (discord bot)
- Create workflows (automatically create Notion meeting notes, etc.)

Eventually, the project will:
- Support **research workflows** in explainable AI (XAI) and neurosymbolic methods
- Serve as a **teaching and mentoring aid** for students
- Enable **controlled experimentation** with reasoning, evaluation, and grounding
- Provide a platform for studying **what models know, don’t know, and how they explain it**

---

## High level overview

1. Wait for Zoom Webhook to fire. Zoom will send events, when a meeting is completed.
2. After an event is sent, the pipeline will capture all audio files from the specific meeting session then transcribe (Whisper) and diarize (Pyannote) them.
3. The diarizations will get matched with transcriptions based on timestamps.
4. Gemma (in-house LM) recieves the merged transcripts and diarizations and gives a brief summary of the meeting.

This project is intended primarily for **internal AIEA lab and academic use**.
