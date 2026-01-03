import json
import sys
import os
import tempfile
from utils.speech_to_text import transcribe_video
from utils.transcript_evaluator import evaluate_transcript


def load_payload(path="data/payload.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_to_temp(video_path):
    """
    Copy video asli ke temporary file.
    """
    ext = video_path.split(".")[-1]
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")

    with open(video_path, "rb") as src:
        temp.write(src.read())

    temp.flush()
    temp.close()
    return temp.name


def cleanup_audio(temp_video_path):
    """
    Hapus file audio hasil extract:
    <temp_video>_audio.wav
    """
    base = os.path.splitext(temp_video_path)[0]
    audio_path = base + "_audio.wav"

    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"[CLEANUP] Deleted extracted audio: {audio_path}")
    else:
        print("[CLEANUP] No extracted audio found.")


def run_evaluation(video_path: str, question_id: int, payload):
    # Whisper transcription
    print("🔄 Running Whisper transcription...")
    transcript_text = transcribe_video(
        video_path,
        prompt="This audio is an English HR interview. Transcribe clearly."
    )

    review_items = payload["data"]["reviewChecklists"]["interviews"]
    item = next((q for q in review_items if q["positionId"] == question_id), None)

    if item is None:
        raise ValueError(f"Question ID {question_id} tidak ditemukan di payload.json")

    question = item["question"]

    # Evaluator
    result = evaluate_transcript(
        question_id=question_id,
        question=question,
        answer=transcript_text
    )

    print("\n=== FINAL Evaluator Output ===")
    print(json.dumps(result, indent=2))
    print("====================================\n")


def main():
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python test_evaluator1.py <video_file.webm> <question_id>\n")
        sys.exit(1)

    input_video = sys.argv[1]
    question_id = int(sys.argv[2])

    if not os.path.exists(input_video):
        print(f"ERROR: File tidak ditemukan -> {input_video}")
        sys.exit(1)

    # Copy ke temp
    print("[INFO] Copying video to temporary file...")
    temp_video = copy_to_temp(input_video)

    payload = load_payload()

    try:
        # Run evaluator
        run_evaluation(temp_video, question_id, payload)

    finally:
        # Always cleanup
        cleanup_audio(temp_video)

        if os.path.exists(temp_video):
            os.remove(temp_video)
            print(f"[CLEANUP] Deleted temp video: {temp_video}")


if __name__ == "__main__":
    main()
