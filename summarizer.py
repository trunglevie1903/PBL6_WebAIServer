import nltk
nltk.download('punkt_tab')
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class Summarizer:
  def __init__(self, model_path):
    # Load the trained DistilBART model and tokenizer
    self.summarize_model = BartForConditionalGeneration.from_pretrained("./model")
    self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.summarize_model.to(self.device)
    
  def split_into_segments(self, text, max_token_len=1024):
    sentences = nltk.tokenize.sent_tokenize(text)
    segments = []
    current_segment = ""
    
    for sentence in sentences:
      # Add sentence to the segment and check token count
      temp_segment = current_segment + " " + sentence
      tokens = self.tokenizer(temp_segment, truncation=False)["input_ids"]
      
      if len(tokens) > max_token_len:
        # If adding another sentence exceeds the token limit, save current segment
        segments.append(current_segment.strip())
        current_segment = sentence  # Start a new segment
      else:
        # Otherwise, add the sentence to the current segment
        current_segment = temp_segment
  
    # Add any remaining text as the last segment
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments
    
  def summarize_segment(self, text, max_length=64, min_length=16, num_beams=4):
    inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: value.to(self.device) for key, value in inputs.items()}
    summary_ids = self.summarize_model.generate(
      inputs['input_ids'],
      max_length=max_length,
      min_length=min_length,
      num_beams=num_beams,
      length_penalty=2.0,
      early_stopping=True
    )
    return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

  # def summarize(self, long_text):
  #   # Step 1: Split text into sentence-based segments
  #   segments = self.split_into_segments(long_text)
    
  #   # Step 2: Summarize each segment individually
  #   segment_summaries = [self.summarize_segment(segment) for segment in segments]
    
  #   # Step 3: (Optional) Summarize the summaries to get an overall summary
  #   combined_summary = " ".join(segment_summaries)
  #   final_summary = self.summarize_segment(combined_summary, max_length=150, min_length=50)
    
  #   return final_summary

  def summarize(self, complete_transcript):
    # Tokenize the input transcript
    inputs = self.tokenizer(complete_transcript, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(self.device) for key, value in inputs.items()}
    # Generate the summary (you can adjust the max_length, num_beams, etc.)
    summary_ids = self.summarize_model.generate(inputs['input_ids'], max_length=64, min_length=16, num_beams=8, length_penalty=1.5, early_stopping=True)
    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(summary)
    # return summary
    
    transcript_words = complete_transcript.split(" ")
    l = len(transcript_words)
    while (l > 500):
      summary = ""
      for i in range(0, (int)(l//400)):
        start_pos = i*400
        end_pos = min(start_pos + 500 , l)
        sub_section = transcript_words[start_pos: end_pos]
        substring = " ".join(sub_section)
        inputs = self.tokenizer(substring, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.summarize_model.generate(inputs['input_ids'], max_length=64, num_beams=8, length_penalty=1.5, early_stopping=True)
        summary += self.tokenizer.decode(summary_ids[0], skip_special_tokens=True) + " "
      transcript_words = summary.split(" ")
      l = len(transcript_words)
    # print(summary)

    inputs = self.tokenizer(summary, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = self.summarize_model.generate(inputs['input_ids'], max_length=150, num_beams=8, length_penalty=2.0, early_stopping=True)
    final_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # print("final summary: ", final_summary)
    return final_summary

  # def summarize(self, complete_transcript, chunk_size=1024, final_summary_length=128):
  #   # Tokenize and split the complete transcript into chunks
  #   input_ids = self.tokenizer(complete_transcript, return_tensors="pt", max_length=chunk_size, truncation=True).input_ids
  #   num_chunks = (input_ids.size(1) // chunk_size) + 1
  #   summaries = []

  #   for i in range(num_chunks):
  #       # Extract chunk and summarize it
  #       chunk_input_ids = input_ids[:, i * chunk_size:(i + 1) * chunk_size].to(self.device)
  #       summary_ids = self.summarize_model.generate(chunk_input_ids, max_length=64, min_length=16, num_beams=4, length_penalty=1.0, early_stopping=False)
  #       chunk_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  #       summaries.append(chunk_summary)

  #   # Optionally, summarize the combined summaries for a final summary
  #   combined_summary = " ".join(summaries)
  #   print(combined_summary)
  #   final_inputs = self.tokenizer(combined_summary, return_tensors="pt", max_length=chunk_size, truncation=True).input_ids.to(self.device)
  #   final_summary_ids = self.summarize_model.generate(final_inputs, max_length=final_summary_length, num_beams=4, length_penalty=1.0, early_stopping=True)
  #   final_summary = self.tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
  #   # print(final_summary)
  #   return final_summary
