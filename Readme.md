# transcribe-conversation-by-speakers


## Motivations:

1. For those preparing for job interviews and watching YouTube for mock interview videos, it is very tiresome and time taking to sit through the length of the video in order to get all the questions asked. Can we get all the questions and answers transcribed instead? The most convenient way was the youtube’s inbuilt transcription feature. However, the conversation was separated by timestamps instead of Person speaking. Is there a way I can distinguish between the candidates and the interviewer?

2. What if you need to transcribe a video(seminar/conference etc) that you do not wish to upload to YouTube due to privacy or compliance concerns?


# Low level implementation and steps:

OS Used - MacOS  
Language - python

 - Whisper Transcription  Shortcut uses OpenAI’s whisper model.  
 - Whisperx is an opensource tool that has been built by implementing whisper and pyannote-audio together.  
 - There are 5 models available - tiny, base, small, medium, large. Smaller models->fast transcription->low accuracy  
 - Make sure you have enough storage for model download. Models are downloaded at /Users/yourname/.cache/whisper  
 - You can customize model location with -> export TORCH_HOME=/custom/path/you/choose  

## Steps:  
0. To save processing time/memory, please convert your video file (mkv, mp4) to audio (mp3, m4a). I have used macos app Shutter Encoder. You can also use terminal command with ffmpeg.

1. Install Homebrew (if you don’t already have it)

2. Install pyenv + pyenv-virtualenv
```c
brew install pyenv pyenv-virtualenv
```

Add these to your shell init (zsh on macOS):
```c
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
source ~/.zshrc
```

3. Install Python 3.9.19 and create a virtualenv
```c
pyenv install 3.9.19
mkdir -p ~/whisperx-project
cd ~/whisperx-project

\# Create a dedicated env
pyenv virtualenv 3.9.19 whisperx-env

\# Make it auto-activate in this folder
pyenv local whisperx-env

Tip: Because you ran pyenv local whisperx-env, just cd-ing into ~/whisperx-project later will auto-activate the env. You don’t need to run pyenv activate every time.
Check:
which python
python --version
```
4. Inside ~/whisperx-project:
```c
mkdir -p input_audios output_transcripts processed_audios
paste the requirements.txt
Install:
pip install --upgrade pip
pip install -r requirements.txt
```

5. Hugging Face access (token + gated-model acceptance)  
Go to: https://huggingface.co/settings/tokens  
Click Create new token  
Select type read  
enter token name such as whisperx-token  
copy the token value
```c
command in terminal : export HUGGINGFACE_TOKEN="your_token_here"
verify if token is working correctly:
python -c "from huggingface_hub import HfApi; print(HfApi(token='YOUR_TOKEN_HERE').whoami())"
or
read -s HF_TOKEN
python -c "from huggingface_hub import HfApi; import os; print(HfApi(token=os.environ['HF_TOKEN']).whoami())"
```
Click "Access repository" and "Agree and access" for below urls
https://hf.co/pyannote/speaker-diarization-3.1  
https://huggingface.co/pyannote/speaker-diarization-3.1s

6. Python script  
Paste batch_whisperx.py in ~/whisperx-project  
What this script does:  
What this script does (as per your working setup):
 - CPU-only WhisperX “medium” model
 - Processes one file at a time (keeps Mac responsive)
 - Limits PyTorch to one thread (torch.set_num_threads(1))
 - Sorts files by smallest first
 - Uses Hugging Face token for pyannote diarization
 - Labels speakers as Person 1/2/3
 - Writes transcripts to output_transcripts/…_transcript.txt
 - Moves processed audio into processed_audios/
 - Prints per-file and total timings

7. Running:
```c
cd ~/whisperx-project
\# pyenv auto-activates whisperx-env because of `pyenv local` earlier

python batch_whisperx.py

To exit venv give command: deactivate
```
8. (Optional) Splitting large MP3s (if diarization stalls on very long files)
```c
brew install ffmpeg
ffmpeg -i bigfile.mp3 -f segment -segment_time 900 -c copy "input_audios/bigfile_part_%03d.mp3"
```
