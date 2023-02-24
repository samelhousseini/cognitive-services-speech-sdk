import logging
import json
import azure.functions as func
import pandas as pd
import requests
import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import smart_open
import openai
import pinecone
import numpy as np
import time

from redis import Redis
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import TagField
from redis.commands.search.query import Query
from redis.commands.search.result import Result

ITEM_KEYWORD_EMBEDDING_FIELD='item_vector'
DAVINCI_EMBED_LEN = 12288
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-search-davinci-doc-001"
EMB_QUERY_MODEL = "text-search-davinci-query-001"

API_KEY = os.environ["API_KEY"]
RESOURCE_ENDPOINT = os.environ["RESOURCE_ENDPOINT"]
BLOB_CONN = os.environ["BLOB_CONN"]
CONTAINER = os.environ["CONTAINER"]
OUTCONTAINER = os.environ["OUTCONTAINER"]
USE_REDIS = os.environ["USE_REDIS"]
REDIS_ADDR = os.environ["REDIS_ADDR"]   
PINECODE_KEY = os.environ["PINECODE_KEY"]
PINECODE_ENV = os.environ["PINECODE_ENV"]   


openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"

TEMPERATURE = 0
MAX_TOKENS = 750



## Re-engineer the below main function
## 1. include "chunking"
## 2. include Cognitive Services translation
## 3. remove embedding matching below
## 4. re-engineer the prompt to achieve multiple tasks at the same time, e.g. summarization, phrases, etc.. (use the below example as guideline)

"""
You must extract the following information from the phone conversation below:

1. Call reason (key: reason)
2. Cause of the incident (key: cause)
3. Names of all drivers as an array (key: driver_names)
4. Insurance number (key: insurance_number)
5. Accident location (key: location)
6. Car damages as an array (key: damages)
7. A short, yet detailed summary (key: summary)

Make sure fields 1 to 6 are answered very short, e.g. for location just say the location name

Please answer in JSON machine-readable format, using the keys from above.Phone conversation:

Hi I just had a car accident and wanted to report it. OK, I hope you're alright, what happened? I was driving on the I-18 and I hit another car. Are you OK? Yeah, I'm just a little shaken up. That's understandable. Can you give me your full name? Sure, it's Sarah standl. Do you know what caused the accident? I think I might have hit a pothole. OK, where did the accident take place? On the I-18 freeway. Was anyone else injured? I don't think so. But I'm not sure. OK, well we'll need to do an investigation. Can you give me the other drivers information? Sure, his name is John Radley. And your insurance number. OK. Give me a minute. OK, it's 546452. OK, what type of damages has the car? Headlights are broken and the airbags went off. Are you going to be able to drive it? I don't know. I'm going to have to have it towed. Well, we'll need to get it inspected. I'll go ahead and start the claim and we'll get everything sorted out. Thank you.

JSON:

"""








def main(msg: func.ServiceBusMessage):

    print("Using Redis", USE_REDIS)
    logging.info(f"Using Redis {USE_REDIS}")

    if USE_REDIS == "1":
        port = 6379
        redis_conn = Redis(host = REDIS_ADDR, port = port)
        logging.info('Connected to redis')

        try:
            out = redis_conn.ft('emb_index').info()
            logging.info(f"Found Redis Index {out}")
        except Exception as e:
            logging.info(e)
            logging.info("Creating Search Index ###########################################################")
            def create_search_index (redis_conn,vector_field_name,number_of_vectors, vector_dimensions=512, distance_metric='L2'):
                M=40
                EF=200
                index_name = "emb_index"
                redis_conn.ft(index_name).create_index([
                    #VectorField(vector_field_name, "FLAT", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "BLOCK_SIZE":number_of_vectors }),
                    VectorField(vector_field_name, "HNSW", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "M": M, "EF_CONSTRUCTION": EF}),
                    TagField("id"),
                    TextField("text"),
                    TextField("summary"),
                    TagField("timestamp")        
                ])
                logging.info("Progress in creating Search Index ###########################################################")
                #time.sleep(5)

            ITEM_KEYWORD_EMBEDDING_FIELD='item_vector'
            DAVINCI_EMBED_LEN = 12288
            NUMBER_PRODUCTS=1000

            #flush all data
            redis_conn.flushall()

            #create flat index & load vectors
            create_search_index(redis_conn, ITEM_KEYWORD_EMBEDDING_FIELD,NUMBER_PRODUCTS,DAVINCI_EMBED_LEN,'COSINE')
            logging.info("Done creating Search Index ###########################################################")

    else:
        pinecone.init(api_key=PINECODE_KEY,environment=PINECODE_ENV)
        if 'contoso' not in pinecone.list_indexes():
            pinecone.create_index('contoso', dimension=DAVINCI_EMBED_LEN)    
        index = pinecone.Index('contoso')


    msg_dict = json.loads(msg.get_body().decode('utf-8'))
    logging.info(" ")
    logging.info(type(msg_dict))
    logging.info(msg_dict)
    logging.info(" ")


    logging.info("Event Type:%s", msg_dict['eventType'])

    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN)
    container_client = blob_service_client.get_container_client(CONTAINER)
    #blob_list = container_client.list_blobs(BLOB_DIR) 

    transport_params = {
    'client': BlobServiceClient.from_connection_string(BLOB_CONN),
    }

    json_filename = os.path.basename(msg_dict['subject'])

    # stream from Azure Blob Storage
    with smart_open.open(f"azure://{CONTAINER}/{json_filename}", transport_params=transport_params) as fin:
        data = json.load(fin)

    data["recognizedPhrases"] = sorted(data["recognizedPhrases"], key=lambda phrase : phrase["offsetInTicks"])
    phrases = get_transcription_phrases(data)
    conversation_items = transcription_phrases_to_conversation_items(phrases)

    conversation = ''
    for item in conversation_items:
        conversation += item['role'] + ': ' + item['text'] + '\n'

    data['conversation'] = conversation
    logging.info(conversation)        

    summary = openai_summarize(conversation)
    print("Summary:\n",summary,'\n')
    data['summary'] = summary

    logging.info(summary)        

    topic = openai_extract_topic(conversation)
    print("Topic:\n",topic,'\n')
    data['topic'] = [t.strip() for t in topic.split('\n')]

    keys = openai_extract_keyphrases(conversation)
    keys = keys.split('\n')
    items = keys[0].split(':')[1].split(',') + keys[1].split(':')[1].split(',')
    print("Keywords and Key Phrases:\n",items,'\n')

    data['key_items'] = items

    ner = openai_extract_ner(conversation)
    print("NER:\n",ner,'\n')
    data['ner'] = [n.strip() for n in ner.split('\n')]

    sentiment = openai_extract_sentiment(conversation)
    print("Sentiment Analysis:\n",sentiment,'\n')
    data['sentiment'] = sentiment

    category = openai_classify_text(conversation)
    print("Category:\n",category,'\n')
    data['category'] = category

    # q = "what is the name of the customer?"
    # answer = openai_interrogate_text(conversation, q)
    # logging.info(f"Text Interrogation: {q}\n", answer,'\n')

    if USE_REDIS == "1":
        redis_upsert_embedding(redis_conn, conversation, summary, data, json_filename)
        print("Upsert Complete")
        res = redis_query_embedding_index(redis_conn, conversation, json_filename)
        print("This is res:", res)
        data['related'] =  res
    else:
        upsert_embedding(index, conversation, summary, data, json_filename)
        print("Upsert Complete")
        res = query_embedding_index(index, conversation, json_filename)
        print("This is res:", res)
        data['related'] =  res

    # Create a blob client using the local file name as the name for the blob
    print("\nUploading to Azure Storage as blob:\n\t" + json_filename)
    blob_client = blob_service_client.get_blob_client(container=OUTCONTAINER, blob=json_filename)
    blob_client.upload_blob(json.dumps(data, indent = 4), overwrite=True)
    



def redis_upsert_embedding(redis_conn, conversation, summary, transcription, transcription_id):
    res = openai.Embedding.create(input=conversation, engine=EMBEDDING_MODEL, deployment_id=EMBEDDING_MODEL)
    embeds = np.array([res['data'][0]['embedding']]).astype(np.float32).tobytes()
    meta = {'text': conversation, 'summary': summary, 'timestamp': transcription['timeStamp'], ITEM_KEYWORD_EMBEDDING_FIELD:embeds}
    p = redis_conn.pipeline(transaction=False)
    p.hset(transcription_id, mapping=meta)
    try:
        p.execute()
    except Exception as e:
        print(e)


def redis_query_embedding_index(redis_conn, query, transcription_id, topK=5):
    xq = openai.Embedding.create(input=query, engine=EMB_QUERY_MODEL)['data'][0]['embedding']
    query_vector = np.array(xq).astype(np.float32).tobytes()
    q = Query(f'*=>[KNN {topK} @{ITEM_KEYWORD_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score')\
                                .paging(0,topK).return_fields('vector_score','summary','text','timestamp').dialect(2)
    params_dict = {"vec_param": query_vector}
    results = redis_conn.ft("emb_index").search(q, query_params = params_dict)
    
    return [{
                'id':match.id , 
                'text':match.text, 
                'summary':match.summary, 
                'timestamp':str(match.timestamp), 
                'score':match.vector_score
            } 
            for match in results.docs if match.id != transcription_id]


def upsert_embedding(index, conversation, summary, transcription, transcription_id):
    res = openai.Embedding.create(input=conversation, engine=EMBEDDING_MODEL, deployment_id=EMBEDDING_MODEL)
    ids = [transcription_id]
    embeds = [res['data'][0]['embedding']]    
    meta = [{'text': conversation, 'summary': summary, 'timestamp': transcription['timeStamp']}]
    print(ids, embeds[0][:2], meta)
    to_upsert = zip(ids, embeds, meta)
    return index.upsert(vectors=list(to_upsert))



def query_embedding_index(index, query, transcription_id):
    xq = openai.Embedding.create(input=query, engine=EMB_QUERY_MODEL)['data'][0]['embedding']
    print(xq[:5])

    res = index.query(
                    [xq], 
                    top_k=5, 
                    include_metadata=True)
    
    return [{
                'id':match['id'] , 
                'text':match['metadata']['text'], 
                'summary':match['metadata']['summary'], 
                'timestamp':str(match['metadata']['timestamp']), 
                'score':match['score']
            } 
            for match in res['matches'] if match['id'] != transcription_id]




def w(fn, s):
    f = open(fn, "a")
    f.write(s)
    f.close()



def contact_openai(prompt):

    return openai.Completion.create(
                    prompt=prompt,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    model=COMPLETIONS_MODEL,
                    deployment_id=COMPLETIONS_MODEL
                )["choices"][0]["text"].strip(" \n")


def openai_summarize(conversation):

    prompt =f"""
    Summarize the following text.

    Text:
    ###
    {conversation}
    ###

    Summary:

    """

    return contact_openai(prompt)


def openai_extract_ner(conversation):

    prompt =f"""
    From the text below, extract the following entities in the following format:
    Companies: <comma-separated list of companies mentioned>
    People & titles: <comma-separated list of people mentioned (with their titles or roles appended in parentheses)>

    Text:
    ###
    {conversation}
    ###

    List:

    """
    return contact_openai(prompt)




def openai_extract_keyphrases(conversation):

    prompt =f"""
    From the text below, extract keywords and keyphrases in the following format:
    Keywords: <comma-separated list of keywords mentioned>
    Key Phrases: <comma-separated list of key phrases>

    Text:
    ###
    {conversation}
    ###

    List:

    """
    return contact_openai(prompt)




def openai_extract_topic(conversation):

    prompt =f"""
    Extract the topic from the below in bullet points format:

    Text:
    ###
    {conversation}
    ###

    Topic bullet point:

    """
    return contact_openai(prompt)


def openai_extract_sentiment(conversation):

    prompt =f"""
    Classify the sentiment in the below:

    Text:
    ###
    {conversation}
    ###

    Sentiment rating:

    """
    return contact_openai(prompt)



def openai_interrogate_text(conversation, question):

    prompt =f"""
    {question}:

    Text:
    ###
    {conversation}
    ###

    Answer:

    """
    return contact_openai(prompt)



def openai_classify_text(conversation):

    prompt =f"""
    Classify the below text into the following categories : [Health, Insurance, Finance, Business, Technology, Agriculture, Mining, Pharmaceutical, Retail, Transportation]

    Text:
    ###
    {conversation}
    ###

    Category:

    """
    return contact_openai(prompt)




class TranscriptionPhrase(object) :
    def __init__(self, id : int, text : str, itn : str, lexical : str, speaker_number : int, offset : str, offset_in_ticks : float) :
        self.id = id
        self.text = text
        self.itn = itn
        self.lexical = lexical
        self.speaker_number = speaker_number
        self.offset = offset
        self.offset_in_ticks = offset_in_ticks



def get_transcription_phrases(transcription) :

    def helper(id_and_phrase ):
        (id, phrase) = id_and_phrase
        best = phrase["nBest"][0]
        speaker_number : int
        # If the user specified stereo audio, and therefore we turned off diarization,
        # only the channel property is present.
        # Note: Channels are numbered from 0. Speakers are numbered from 1.
        # if "speaker" in phrase :
        #     speaker_number = phrase["speaker"] - 1
        # el
        if "channel" in phrase :
            speaker_number = phrase["channel"]
        else :
            raise Exception(f"nBest item contains neither channel nor speaker attribute.{linesep}{best}")

        return TranscriptionPhrase(id, best["display"], best["itn"], best["lexical"], speaker_number, phrase["offset"], phrase["offsetInTicks"])

    # For stereo audio, the phrases are sorted by channel number, so resort them by offset.
    return list(map(helper, enumerate(transcription["recognizedPhrases"])))


def transcription_phrases_to_conversation_items(phrases) :
    return [{
        "id" : phrase.id,
        "text" : phrase.text,
        "itn" : phrase.itn,
        "lexical" : phrase.lexical,
        # The first person to speak is probably the agent.
        "role" : "Agent" if 0 == phrase.speaker_number else "Customer",
        "participantId" : phrase.speaker_number
    } for phrase in phrases]    