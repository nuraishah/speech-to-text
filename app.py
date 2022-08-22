from model import *
import librosa
from pathlib import Path

### Option 1
def record_voice():
    audiorec_demo_app()

### Option 2
def upload_trans():
    # Upload file 
    uploaded_file = st.file_uploader("Select file from your directory", type=".wav")

    if uploaded_file:
        filename = uploaded_file.name
        path = "./tmp" #Define path where the files are stored
        save_uploadedfile(uploaded_file, filename, path)

        # Calling back the file and load it to get numpy arrays
        save_path = Path('./tmp') / filename
        audio, rate = librosa.load(save_path, sr = 16000)
        return audio

def start_transribe(audio):
    if st.button("Transcribe"):
        # Importing Wav2Vec pretrained model
        st.write("Text:")
        transcription = transribe_audio(audio)
        st.write(transcription)

st.sidebar.markdown("""
### You can start transcribing by:
1. Uploading wav file  
1. Recording your voice
""")
start_op = st.sidebar.selectbox('Select here:', ('Upload wav file', 'Record voice'))

if start_op == 'Upload wav file':
    st.write("1. Upload wav file")
    audio = upload_trans()
    st.write("2. Start transcribe")
    start_transribe(audio)

elif start_op == 'Record voice':
    st.write("1. Record voice")
    record_voice()
    st.write("2. Upload wav file")
    audio = upload_trans()
    st.write("3. Start transcribing")
    start_transribe(audio)