from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en")
transcript_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarize_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")

file_mp3_path = "./reacting to the world biggest house.mp3"
audio, sr = librosa.load(file_mp3_path, sr=16000)
sr = 16000  # Sample rate used in the Whisper model
chunk_length_s = 30
chunk_length = int(chunk_length_s * sr)
audio_chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
transcript_parts = []
l = len(audio_chunks)
for i, chunk in enumerate(audio_chunks):
    print(f"{i+1}/{l}")
    input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = transcript_model.generate(input_features)
    # Decode the predicted ids to get the transcription of the chunk
    chunk_transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    transcript_parts.append(chunk_transcript)

# Combine all chunk transcripts into a complete transcript
complete_transcript = ' '.join(transcript_parts)

transcript_words = complete_transcript.split(" ")
l = len(transcript_words)
while (l > 512):
  summary = ""
  for i in range(0, (int)(l/400)):
    start_pos = i*400
    end_pos = min((i+1)*400 - 1 , l)
    sub_section = transcript_words[start_pos: end_pos]
    substring = " ".join(sub_section)
    inputs = tokenizer(substring, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarize_model.generate(inputs['input_ids'], max_length=64, min_length=32, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary += tokenizer.decode(summary_ids[0], skip_special_tokens=True) + " "
  transcript_words = summary.split(" ")
  l = len(transcript_words)
print(summary)

inputs = tokenizer(summary, return_tensors="pt", max_length=512, truncation=True)
summary_ids = summarize_model.generate(inputs['input_ids'], max_length=64, min_length=32, num_beams=4, length_penalty=2.0, early_stopping=True)
final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("final summary: ", final_summary)