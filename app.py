import os
import uuid
import threading
import requests  # To send the POST request
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from summarizer import Summarizer
from generate_transcript import TranscriptGenerator

app = Flask(__name__)

# Temporary folder to store mp3 files
TEMP_DIR = "./temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize the Summarizer and TranscriptGenerator
transcriptor = TranscriptGenerator()
summarizer = Summarizer(model_path="./model")

# Function to process the file asynchronously
def process_audio(file_path, video_id):
    try:
        # print(video_id)
        # Generate transcript
        complete_transcript, timestamped_transcripts = transcriptor.generate_transcript(file_path)
        print("Transcript length: ", len(complete_transcript))   

        # Summarize the transcript
        summary = summarizer.summarize(complete_transcript)
        print("Summary:", summary)

        # Send POST request to the webapp backend
        payload = {
            "transcript": timestamped_transcripts,
            "videoId": video_id
        }
        response = requests.post("http://localhost:4000/video/update-transcript", json=payload)
        
        # Check the response from the webapp backend
        if response.status_code == 200:
            print("Successfully updated transcript for videoId:", video_id)
        else:
            print(f"Failed to update backend. Status code: {response.status_code}")
            
        payload = {
            "summary": summary,
            "videoId": video_id
        }
        response = requests.post("http://localhost:4000/video/update-summary", json=payload)
        # Check the response from the webapp backend
        if response.status_code == 200:
            print("Successfully updated summary for videoId:", video_id)
        else:
            print(f"Failed to update backend. Status code: {response.status_code}")

        # Optional: remove the file after processing
        os.remove(file_path)

    except Exception as e:
        print(f"Error processing audio: {e}")


@app.route('/summarize', methods=['POST'])
def summarize():
    # Extract the 'videoId' and the 'file' from the request
    video_id = request.form.get('videoId')
    if not video_id:
        return jsonify({"error": "No videoId provided"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file_mp3 = request.files["file"]
    if file_mp3.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Secure the filename
    original_filename = secure_filename(file_mp3.filename)
    
    # Generate a UUID and create a unique filename
    unique_id = str(uuid.uuid4())  # Generate a random UUID
    file_extension = original_filename.split('.')[-1]  # Get the file extension (e.g., mp3)
    unique_filename = f"{unique_id}.{file_extension}"  # Construct the unique filename

    # Save the file to the temporary folder with the unique filename
    temp_path = os.path.join(TEMP_DIR, unique_filename)
    file_mp3.save(temp_path)

    # Start a new thread to process the file and pass the videoId
    threading.Thread(target=process_audio, args=(temp_path, video_id)).start()

    # Return an immediate response to the client
    response = {
        "message": "Audio file received and processing has started.",
        "file_id": unique_id,  # Return the UUID for reference
        "video_id": video_id   # Return the videoId
    }
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
