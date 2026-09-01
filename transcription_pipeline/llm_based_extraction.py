from agents import Agent, Runner
from pydantic import BaseModel, Field

# fine grained topic segmentation
""" segment interviews into smaller blocks based on topic shifts"""

# for llm based extract, deploy a pydantic dataclass to define detailed extraction guidelines for content specifications and format requirements.

# the extract is checked against the input. if there is no overlap, the value is considered a hallucination and is discarded.

instructions = """You are a meeting segmentation agent. Your job is to split the entire meeting into sections based on topic changes
Return a json format
"""




class TeamMemberUpdates(BaseModel):
    person_name: str = Field(description = "The name of the person responsible for task/ project.")
    project_name: str = Field(description = "The project they are assigned to (if stated)")
    accomplishments: str = Field(description = "What they have accomplished")
    to_do: str = Field(description="What are they planning to do next")
    blockers: str = Field(description="their list of blockers, if any")

class MemberList(BaseModel):
    list_of_persons: list[TeamMemberUpdates] = Field(description = "A list of all topics that were discussed")




agent = Agent(name = "Meeting Segmentation Agent", instructions=instructions, output_type=MemberList)

sum1 = """
(0.0, 3.52)
START
{'timestamp': (0.0, 3.52), 'text': " okay where's the reporting soon", 'speaker': 'SPEAKER_08'}
{'timestamp': (3.52, 11.96), 'text': " i know zoom has changed a lot of stuff so yeah i usually don't use it as a host so", 'speaker': 'SPEAKER_04'}
{'timestamp': (11.96, 27.64), 'text': ' so', 'speaker': 'SPEAKER_09'}
{'timestamp': (30.0, 45.0), 'text': ' Okay, um, Vincent, did you get to start reporting on your side, or?', 'speaker': 'SPEAKER_04'}
{'timestamp': (45.0, 47.0), 'text': ' Yeah, I did.', 'speaker': 'SPEAKER_08'}
{'timestamp': (47.0, 49.0), 'text': ' Okay.', 'speaker': 'SPEAKER_04'}
{'timestamp': (49.0, 51.0), 'text': ' Cool.', 'speaker': 'SPEAKER_05'}
{'timestamp': (51.0, 113.78), 'text': " I don't know, let's see who joined.  Okay.  Why don't we get started? So again, I mentioned this will kind of be, this will likely be a  quicker meeting than previous weeks. So a lot of people said they're busy with midterms and  other things. And a lot of people are out of town, which I'm not sure why. Anyway, we'll keep it  fairly, we'll keep it fairly quick. And I know I was gone last week, so I was going through  everybody's update so thanks for doing that um so let's get started um so a couple things i'm  trying to figure out remember who is the point person for hosting um i guess it's tanya who's  not here but i know um jung way you're on that team um and sergio uh i think is also on that ", 'speaker': 'SPEAKER_04'}
{'timestamp': (113.78, 120.04), 'text': ' team with Benjamin. So maybe we can hear from you guys how things are going. And then I can', 'speaker': 'SPEAKER_04'}
{'timestamp': (120.04, 150.26), 'text': " talk about a little bit about the domain. Yeah, for me, I summarize the docs for my  agentech pack plan for some others onboarding. And I'm waiting to be assigned the task  for the hosting  and I like  to know a little about  the domain so  that's for my ", 'speaker': 'SPEAKER_02'}
{'timestamp': (150.26, 151.86), 'text': ' updates.', 'speaker': 'SPEAKER_02'}
{'timestamp': (153.36, 157.94), 'text': ' Great.  Am I right, Sergio and Benjamin,  you guys were also on ', 'speaker': 'SPEAKER_04'}
{'timestamp': (157.94, 159.08), 'text': ' RAG and hosting?', 'speaker': 'SPEAKER_04'}
{'timestamp': (161.26, 191.58), 'text': " Yeah.  Last week,  I pretty much  just kind of read through  the like  summarized documents that  Zhangvi posted  and then this week I was  planning to get  like  start running this stuff and just  like see how that  goes  I just wasn't able to last week because  I had my turn  and I had my turn tomorrow too ", 'speaker': 'SPEAKER_01'}
STOP
START
{'timestamp': (161.26, 191.58), 'text': " Yeah.  Last week,  I pretty much  just kind of read through  the like  summarized documents that  Zhangvi posted  and then this week I was  planning to get  like  start running this stuff and just  like see how that  goes  I just wasn't able to last week because  I had my turn  and I had my turn tomorrow too ", 'speaker': 'SPEAKER_01'}
{'timestamp': (191.58, 191.96), 'text': ' so', 'speaker': 'SPEAKER_01'}
{'timestamp': (191.96, 203.72), 'text': " I'm a little bit behind  because i had a midterm on thursday so i wasn't able to meet up um with the group so ", 'speaker': 'SPEAKER_07'}
{'timestamp': (203.72, 208.94), 'text': " and i joined on week three so i think i'm planning to ask for a little bit of guidance today", 'speaker': 'SPEAKER_07'}
{'timestamp': (208.94, 214.0), 'text': ' yeah we can definitely talk about that i mean i think then have you been able to go through', 'speaker': 'SPEAKER_04'}
{'timestamp': (214.0, 219.4), 'text': " john way's docs no i haven't that was that was what i was gonna ask okay i think that's your", 'speaker': 'SPEAKER_07'}
{'timestamp': (219.4, 256.1), 'text': " first step um so let's get let's kind of set your next task to be uh to read through the docs and  then sergio will have you actually running the code base and jung way um i have a couple ideas  and maybe you can tell me kind of where where you're at and what you think i mean so we have  this domain registered it's on squarespace and what i'm trying to understand is how we set up a  server for it. So what I would be curious is, I think somewhere we have a Google Cloud backend, ", 'speaker': 'SPEAKER_04'}
{'timestamp': (256.1, 259.88), 'text': ' but maybe for the week you can kind of look into how do we actually set that up?', 'speaker': 'SPEAKER_04'}
{'timestamp': (265.54, 275.44), 'text': " Yes, I can look up something about how to set up the backend on the Google Cloud. I don't know ", 'speaker': 'SPEAKER_02'}
{'timestamp': (275.44, 282.18), 'text': ' very clearly either so uh yeah yeah no absolutely', 'speaker': 'SPEAKER_02'}
{'timestamp': (282.18, 314.1), 'text': " cool um and i can i can talk to you maybe i'll post on the on the channel a little bit john way  because i'm i think that's going to be a joint effort but maybe we can kind of figure out i  think our goal is to kind of figure out what the next tasks are with that yeah okay um anybody else  from um from rag and hosting here i'm also on the team oh yeah vincent where are things i i know ", 'speaker': 'SPEAKER_04'}
{'timestamp': (314.1, 319.86), 'text': " you've been working mostly on the hybrid note stuff but yeah where are not hybrid notes is it", 'speaker': 'SPEAKER_04'}
{'timestamp': (319.86, 329.14), 'text': " right oh sorry but yeah uh where are things with you uh i haven't done a lot but i i skimmed through ", 'speaker': 'SPEAKER_08'}
{'timestamp': (329.14, 339.24), 'text': " bit of the pro slm paper and i've been debugging my own uh rag pipeline okay great um", 'speaker': 'SPEAKER_08'}
{'timestamp': (339.24, 348.22), 'text': ' yeah sounds good um and you feel what are your kind of next steps with that', 'speaker': 'SPEAKER_04'}
STOP
START
{'timestamp': (339.24, 348.22), 'text': ' yeah sounds good um and you feel what are your kind of next steps with that', 'speaker': 'SPEAKER_04'}
{'timestamp': (348.22, 376.52), 'text': " i guess keep debugging uh yeah i guess keep debugging for now because i want to like reduce  the response time because it's it's almost like a minute until the until like you see like a  response from the pipeline yeah and i guess after that i have to wait for the next steps because i'm  not really sure on what to do in uh in uh collaborating on implementing the whole thing ", 'speaker': 'SPEAKER_08'}
{'timestamp': (376.52, 382.46), 'text': ' onto the website yeah okay sounds good', 'speaker': 'SPEAKER_08'}
{'timestamp': (382.46, 391.28), 'text': ' anybody else from rag or any questions from the rag i guess the rag and hosting team any questions ', 'speaker': 'SPEAKER_04'}
{'timestamp': (391.28, 401.52), 'text': ' okay do we have anybody from abductive reasoning hugo it might just be you', 'speaker': 'SPEAKER_04'}
{'timestamp': (401.52, 411.28), 'text': ' oh yeah uh right now uh jonathan is uh going to implement uh link link graph into the code so ', 'speaker': 'SPEAKER_00'}
{'timestamp': (411.28, 419.04), 'text': " right now me and sabrina are looking at link chain and link graph okay okay i'm also here i'm from", 'speaker': 'SPEAKER_00'}
{'timestamp': (419.04, 427.2), 'text': " adopt adoptive too oh yeah yes how are things for you yeah um unfortunately i wasn't able to get that", 'speaker': 'SPEAKER_04'}
{'timestamp': (427.2, 444.88), 'text': ' much work done for the last the last week because of midterms and i have uh from last week and i  also have this week tomorrow as well but i did briefly like look over the learning loop i think  there were like images posted of that on like the google doc and i was like just briefly looking at ', 'speaker': 'SPEAKER_05'}
{'timestamp': (444.88, 450.06), 'text': ' it and hopefully like this week i look more into detail and then figure out like how to kind of', 'speaker': 'SPEAKER_05'}
{'timestamp': (450.06, 456.84), 'text': " get started with it okay yeah and that's of um and uh remind me what that doc is of", 'speaker': 'SPEAKER_04'}
{'timestamp': (456.84, 464.98), 'text': " um it's just called abductive learning notes it's made by jonathan um and chugo and ", 'speaker': 'SPEAKER_05'}
{'timestamp': (464.98, 472.56), 'text': " for sabrina um they have like all their old notes and resources on there yeah yeah i think that's", 'speaker': 'SPEAKER_05'}
{'timestamp': (472.56, 485.44), 'text': " great and also chanchal i think through that process kind of for your next steps make sure  you kind of keep track of like what's clear in that and what's not clear um because hopefully  that's something that we can pass on to other people so if you have questions do you know ", 'speaker': 'SPEAKER_04'}
{'timestamp': (485.44, 496.9), 'text': " write those down and keep track of that so yeah yes that's it cool um anything else from abductive", 'speaker': 'SPEAKER_04'}
{'timestamp': (496.9, 523.94), 'text': " um lm law i think that's just yeah um yeah so i train i train or i ran fine tuning on  two other base models and right now i'm using one of the quen models um and yeah i've gotten like ", 'speaker': 'SPEAKER_03'}
STOP
START
{'timestamp': (496.9, 523.94), 'text': " um lm law i think that's just yeah um yeah so i train i train or i ran fine tuning on  two other base models and right now i'm using one of the quen models um and yeah i've gotten like ", 'speaker': 'SPEAKER_03'}
{'timestamp': (523.94, 533.0), 'text': " the best results of like so far like um like 98 like valid query syntax oh wow really wait that's", 'speaker': 'SPEAKER_03'}
{'timestamp': (533.0, 541.8), 'text': " that's huge yeah but but what are you doing like working at like 50 a couple weeks ago uh well", 'speaker': 'SPEAKER_04'}
{'timestamp': (541.8, 559.4), 'text': ' executability was pretty terrible but this time i got like 84 executability okay but the answer  match like between the generated query running on the generated knowledge base versus like the  ground truth query being on the ground truth knowledge base. Yeah, so like that answer ', 'speaker': 'SPEAKER_03'}
{'timestamp': (559.4, 566.76), 'text': " match is like only 18%. So that's the last thing I have to do. Okay. But yeah. Sounds good. What", 'speaker': 'SPEAKER_03'}
{'timestamp': (566.76, 572.04), 'text': ' do you think is next for you trying to get that 18% up I suppose? Yeah, because I have a good like', 'speaker': 'SPEAKER_04'}
{'timestamp': (572.04, 578.2), 'text': " syntax, valid syntax and good executability. So once you get the answer match up, I don't think ", 'speaker': 'SPEAKER_03'}
{'timestamp': (578.2, 588.64), 'text': " think there's anything else to like um look at so okay okay um how's alex doing i'm not sure okay", 'speaker': 'SPEAKER_03'}
{'timestamp': (588.64, 603.82), 'text': " i will um i'll catch up with alex make sure so i know he has thesis stuff and other things  anybody else from llm law i don't think anybody else is here ", 'speaker': 'SPEAKER_04'}
{'timestamp': (603.82, 615.36), 'text': " yeah okay um alelo i think vincent that's you and i don't know who else is here but", 'speaker': 'SPEAKER_04'}
{'timestamp': (615.36, 654.7), 'text': " yeah how are things with you vincent yeah so i got the uh the multimodal uh transcription model  trained oh okay cool yeah um so currently the the accuracy for it is at like 70 percent  uh because i haven't i haven't done uh enough iterations for the training yet so i think i'll  need like uh better resources to do it quickly um and then the next thing i need to do is what  the recording the this meeting recording i have to fine tune the model because um the model is ", 'speaker': 'SPEAKER_08'}
{'timestamp': (654.7, 662.82), 'text': ' trained on uh on a uh a subset of speakers from a corpus that i found online so oh okay', 'speaker': 'SPEAKER_08'}
{'timestamp': (662.82, 689.48), 'text': " Okay. Let me know how I can help with that. So I've been recording all of our meetings  this quarter. I haven't been using ReadAI for a multitude of different reasons. So I have all of  that data. If there's a place I can put that, let me know. Maybe we can make a Google Drive and I  can upload a bunch of that data for you. Of like the meeting recordings? ", 'speaker': 'SPEAKER_04'}
STOP
START
{'timestamp': (662.82, 689.48), 'text': " Okay. Let me know how I can help with that. So I've been recording all of our meetings  this quarter. I haven't been using ReadAI for a multitude of different reasons. So I have all of  that data. If there's a place I can put that, let me know. Maybe we can make a Google Drive and I  can upload a bunch of that data for you. Of like the meeting recordings? ", 'speaker': 'SPEAKER_04'}
{'timestamp': (689.48, 689.72), 'text': ' Yeah.', 'speaker': 'SPEAKER_04'}
{'timestamp': (690.1, 692.56), 'text': ' Okay. Yeah, that would be really helpful, actually.', 'speaker': 'SPEAKER_08'}
{'timestamp': (692.82, 739.62), 'text': " Okay, let me put that kind of as actions, and maybe for other people, too, that record,  I'll make a Google Drive of meeting recordings for fine-tuning.  Okay, cool.  And then other people can put theirs in, too.  Trying to see, is anybody else from Alelo here?  I guess Salaam's not here.  I should.  Okay.  Cool.  And then so you feel like you have next steps? ", 'speaker': 'SPEAKER_04'}
{'timestamp': (739.62, 741.22), 'text': ' I guess your fine-tuning is your next step.', 'speaker': 'SPEAKER_04'}
{'timestamp': (741.66, 743.14), 'text': ' Yeah, next steps, yeah. ', 'speaker': 'SPEAKER_08'}
{'timestamp': (743.14, 744.66), 'text': ' Those are my next steps.', 'speaker': 'SPEAKER_08'}
{'timestamp': (745.24, 757.3), 'text': ' Okay.  And I know the team went from a lot to a little,  but, yeah, we should probably...  I mean, I know the tasks are now cleaned out, but maybe we can be a little bit more visionary with some of the stuff. ', 'speaker': 'SPEAKER_04'}
{'timestamp': (757.3, 759.44), 'text': " I'll start putting some other tasks on there.", 'speaker': 'SPEAKER_04'}
{'timestamp': (759.96, 760.18), 'text': ' Okay.', 'speaker': 'SPEAKER_03'}
{'timestamp': (763.5, 773.8), 'text': " Cool.  Hybrid notes.  So I think that's Gari, AJ, and Shreepad.  And I know we just met, but how are things going with you guys? ", 'speaker': 'SPEAKER_04'}
{'timestamp': (773.8, 775.12), 'text': ' Or we met, I guess, Thursday.', 'speaker': 'SPEAKER_04'}
{'timestamp': (777.0, 779.96), 'text': " I think for me, it's going good.", 'speaker': 'SPEAKER_06'}
{'timestamp': (780.1, 779.96), 'text': ''}
{'timestamp': (784.28, 791.64), 'text': " I am right now, I haven't started testing, but I'm going to start testing with the margins  and the text written on margin today with Adobe API. ", 'speaker': 'SPEAKER_06'}
{'timestamp': (791.64, 792.92), 'text': " Okay. Wait, remind me which API you're working on?", 'speaker': 'SPEAKER_06'}
{'timestamp': (793.8, 799.96), 'text': ' Adobe API. Okay. Yeah. ', 'speaker': 'SPEAKER_04'}
{'timestamp': (799.96, 804.84), 'text': ' Sounds great. And you have everything that you need. I guess the API setup and all of that.', 'speaker': 'SPEAKER_04'}
{'timestamp': (804.84, 805.88), 'text': ' Yes. ', 'speaker': 'SPEAKER_06'}
{'timestamp': (805.88, 807.24), 'text': ' Great.', 'speaker': 'SPEAKER_06'}
{'timestamp': (809.24, 807.24), 'text': ''}
{'timestamp': (814.24, 895.84), 'text': " A.J. and Shripad, what about you two? Yeah, so we had decided to take a couple of days of doing some research to understand  the question that we're trying to solve, and then we had met earlier today to kind of see  what approaches we're looking at taking in order to figure out if it's a scanned document  or not.  what we understand is we're trying to distinguish between a digitally created pdf versus a scanned  pdf document um and both me and shreepat had different approaches on um identifying mine was  more taking a like just like a straight up script route of look scanning through each page of the  document and basically seeing how much of that page is text versus how much that page is an  embedded image wrapped in a PDF and then setting some sort of threshold of if the entire PDF is  like less than one third of its text versus images, then it's probably going to be a scanned  or some sort of other sort of like deterministic approach like that. Shreepad was, and feel free  to step in if I get it wrong, but Shreepad was interested in taking like a VLM and taking that  document and basically give it to that and getting back a yes or no or even if  we had to do some sort of convolutional neural network or something along those ", 'speaker': 'SPEAKER_09'}
STOP
START
{'timestamp': (814.24, 895.84), 'text': " A.J. and Shripad, what about you two? Yeah, so we had decided to take a couple of days of doing some research to understand  the question that we're trying to solve, and then we had met earlier today to kind of see  what approaches we're looking at taking in order to figure out if it's a scanned document  or not.  what we understand is we're trying to distinguish between a digitally created pdf versus a scanned  pdf document um and both me and shreepat had different approaches on um identifying mine was  more taking a like just like a straight up script route of look scanning through each page of the  document and basically seeing how much of that page is text versus how much that page is an  embedded image wrapped in a PDF and then setting some sort of threshold of if the entire PDF is  like less than one third of its text versus images, then it's probably going to be a scanned  or some sort of other sort of like deterministic approach like that. Shreepad was, and feel free  to step in if I get it wrong, but Shreepad was interested in taking like a VLM and taking that  document and basically give it to that and getting back a yes or no or even if  we had to do some sort of convolutional neural network or something along those ", 'speaker': 'SPEAKER_09'}
{'timestamp': (895.84, 903.92), 'text': ' lines to identify it so yeah which one to choose sorry good no I think that', 'speaker': 'SPEAKER_09'}
{'timestamp': (903.92, 906.94), 'text': ' sounds great I mean if you guys want to split it between you I mean AJ with kind ', 'speaker': 'SPEAKER_04'}
{'timestamp': (906.94, 913.12), 'text': " of a script-based approach what are you thinking of using as the interpreter I'm", 'speaker': 'SPEAKER_04'}
{'timestamp': (913.12, 939.92), 'text': " I'm not 100% sure about how I'm going to implement it.  That's the thing.  What does it mean for a page to be like 30% text?  Because a lot of digitally created PDFs that are maybe made in Google can have images inside of it that's like a chart or something.  So how do you even determine that?  So that requires a lot more thought on my end in order to see.  But I really like Shreepad's idea of just giving it to a VLM. ", 'speaker': 'SPEAKER_09'}
{'timestamp': (939.92, 942.46), 'text': ' The more I kind of think it out, it may be more practical.', 'speaker': 'SPEAKER_09'}
{'timestamp': (943.12, 967.72), 'text': " And and if that's what you guys want to do, I mean, both of you can work on that approach and then you guys can both try different VLMs.  I know Jonathan wrote on the channel. What was it? Gemini?  Yeah, he did say Gemma 4. He said he said Gemma 4, like maybe one of you guys could try that.  I don't know. I haven't played with it, but these things are changing every day.  So, I mean, each of you could choose a VLM to test with and see how it does. ", 'speaker': 'SPEAKER_04'}
{'timestamp': (967.72, 969.62), 'text': " But it's up to you guys. So.", 'speaker': 'SPEAKER_04'}
{'timestamp': (970.96, 972.18), 'text': ' OK, sweet. Sounds good.', 'speaker': 'SPEAKER_09'}
{'timestamp': (973.12, 1002.32), 'text': " Yep, I think that sounds good. Yeah. Okay. And Vincent posted something.  Oh, cool. Yeah, that's just the data I was using. Wow, that's huge.  But AJ, Shreepa, do you guys feel like you have next steps? So I think for you guys,  if you want to work on the purely VLM approach, I think that's great. ", 'speaker': 'SPEAKER_04'}
STOP
"""


output = Runner.run_sync(starting_agent = agent, input = f"segment this meeting {sum1}").final_output

print(output)