# import wave
# import subprocess
# import time
# from pathlib import Path


# VOICE = "D:\\EduCast-AI\\voices\\en_US-lessac-medium.onnx"


# def text_to_audio(text):

#     filename = f"audio_{int(time.time())}.wav"

#     subprocess.run(
#         [
#             "piper",
#             "--model",
#             VOICE,
#             "--output_file",
#             filename,
#         ],
#         input=text.encode("utf-8"),
#         check=True,
#     )

#     return filename

from gtts import gTTS
import time

def text_to_audio(text):

    filename = f"audio_{int(time.time())}.mp3"

    tts = gTTS(
        text=text,
        lang="en"
    )

    tts.save(filename)

    return filename