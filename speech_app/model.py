import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import torch
import streamlit.components.v1 as components

def audiorec_demo_app():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)
    st_audiorec()

def save_uploadedfile(uploadedfile, file_name, path):
    '''
    Save uploaded file to path directory
    '''
    with open(os.path.join(path,file_name),"wb") as f:
        f.write(uploadedfile.getbuffer())

def transribe_audio(audio):
    tokenizer = Wav2Vec2Processor.from_pretrained('mesolitica/wav2vec2-xls-r-300m-mixed')
    model = Wav2Vec2ForCTC.from_pretrained('mesolitica/wav2vec2-xls-r-300m-mixed')
    input_values = tokenizer(audio, return_tensors = "pt", sr = 16000).input_values    
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim = -1)
    transcription = tokenizer.batch_decode(prediction)[0]    
    return transcription