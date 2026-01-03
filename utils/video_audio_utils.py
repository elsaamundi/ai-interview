import subprocess
import os

def extract_audio(video_path, output_audio_path=None):
    """
    Extract audio from video using ffmpeg with Whisper-friendly parameters:
    - 16 kHz
    - mono
    - PCM S16LE format
    """

    if output_audio_path is None:
        base = os.path.splitext(video_path)[0]
        output_audio_path = base + "_audio.wav"

    # ffmpeg command
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                     # no video
        "-acodec", "pcm_s16le",    # PCM signed 16-bit little-endian
        "-ar", "16000",            # 16 kHz sample rate
        "-ac", "1",                # mono
        output_audio_path,
        "-y"                       # overwrite without asking
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio using ffmpeg:\n{e.stderr.decode()}")

    return output_audio_path
