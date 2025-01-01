from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
import os

class TranscriptGenerator:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en")
        self.transcript_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    def generate_transcript(self, mp3_file):
        mp3_file_name = mp3_file.split(".")[1].split('\\')[-1]
        
        audio, sr = librosa.load(mp3_file, sr=16000)
        sr = 16000  # Sample rate used in the Whisper model
        chunk_length_s = 20  # Each chunk is 30 seconds long
        chunk_length = int(chunk_length_s * sr)
        audio_chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
        transcript_parts = []
        timestamped_transcripts = []
        
        l = len(audio_chunks)
        for i, chunk in enumerate(audio_chunks):
            print(f"{mp3_file_name}: chunk number {i+1}/{l}")
            input_features = self.processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                predicted_ids = self.transcript_model.generate(input_features)
            
            # Decode the predicted ids to get the transcription of the chunk
            chunk_transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcript_parts.append(chunk_transcript)

            # Calculate the start timestamp of this chunk (in seconds)
            start_timestamp = (i * chunk_length_s)
            
            # Store the transcript chunk with the start timestamp
            timestamped_transcripts.append({"timestamp": start_timestamp, "text": chunk_transcript})

        # Combine all chunk transcripts into a complete transcript
        complete_transcript = ' '.join(transcript_parts)
        if os.path.exists(mp3_file_name):
            os.remove(mp3_file_name)
        
        return complete_transcript, timestamped_transcripts
