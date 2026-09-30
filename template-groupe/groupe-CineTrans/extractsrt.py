import whisper 
import ffmpeg #you have to install ffmpeg-python not jsut ffmepg

ffmpeg.input('videos/input.mp4').output('output.wav').run()

model = whisper.load_model("base")
result = model.transcribe("output.wav")
print(result)

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def transcript_to_srt(transcript):
    srt_output = ""
    for i, segment in enumerate(transcript['segments']):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        srt_output += f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_output

srt_content = transcript_to_srt(result)
with open("output.srt", "w", encoding="utf-8") as srt_file:
    srt_file.write(srt_content)