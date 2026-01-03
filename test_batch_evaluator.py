import os
import json
import tempfile
from utils.speech_to_text import transcribe_video
from utils.transcript_evaluator import evaluate_transcript
from utils.eye_focus_detection import process_video_for_gaze


# =============================
# LOAD PAYLOAD
# =============================
def load_payload(path="data/payload.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================
# COPY VIDEO TO TEMP
# =============================
def copy_to_temp(video_path):
    ext = video_path.split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")

    with open(video_path, "rb") as src:
        tmp.write(src.read())

    tmp.flush()
    tmp.close()
    return tmp.name


# =============================
# CLEANUP EXTRACTED AUDIO
# =============================
def cleanup_audio(temp_video_path):
    base = os.path.splitext(temp_video_path)[0]
    audio_path = base + "_audio.wav"

    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"[CLEANUP] Deleted extracted audio: {audio_path}")


# =============================
# PROCESS SINGLE VIDEO
# =============================
def process_single_video(temp_video_path, question_id, payload, video_id):
    # === TRANSCRIBE ===
    print(f"🎙️  Transcribing: {temp_video_path}")
    transcript_text = transcribe_video(
        temp_video_path,
        prompt="This audio is an English HR interview. Transcribe clearly."
    )

    # === GET QUESTION ===
    items = payload["data"]["reviewChecklists"]["interviews"]
    item = next((q for q in items if q["positionId"] == question_id), None)
    if item is None:
        raise ValueError(f"Question ID {question_id} tidak ditemukan di payload.json")
    question = item["question"]

    # === TRANSCRIPT EVALUATION ===
    eval_result = evaluate_transcript(
        question_id=question_id,
        question=question_text,
        answer=transcript_text
    )

    # === EYE FOCUS ANALYSIS ===
    print(f"👀  Eye Focus: {temp_video_path}")
    try:
        gaze = process_video_for_gaze(temp_video_path)
    except Exception as e:
        gaze = {
            "status": "error",
            "error": str(e)
        }

    # === FINAL OUTPUT JSON ===
    return {
        "id": video_id,
        "video": temp_video_path,
        "transcript": transcript_text,
        "evaluation": {
            "score": eval_result.get("score"),
            "similarity": eval_result.get("similarity"),
            "reason": eval_result.get("reason"),
            "relevance": eval_result.get("relevance", None)
        },
        "eye_focus": (
            {
                "focus_percentage": gaze.get("focus_percentage"),
                "left_glance_percentage": gaze.get("left_glance_percentage"),
                "right_glance_percentage": gaze.get("right_glance_percentage"),
                "suspicious_event_count": gaze.get("suspicious_event_count"),
                "summary_note_cv": gaze.get("summary_note_cv"),
                "raw": gaze
            }
            if gaze and gaze.get("status") == "success"
            else {"status": "failed", "error": gaze.get("error", "unknown")}
        )
    }



# =============================
# MAIN BATCH LOGIC
# =============================
def batch_evaluate(folder_path, payload):
    supported_ext = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    results = []

    video_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_ext
    ]

    if not video_files:
        print("❌ Tidak ada file video ditemukan di folder tersebut.")
        return []

    print(f"📦 Ditemukan {len(video_files)} video.")

    for idx, filename in enumerate(video_files, start=1):
        video_path = os.path.join(folder_path, filename)

        # Extract question_id dari nama file
        if "question_" not in filename.lower():
            print(f"⚠️ Dilewati (tidak ada ID di nama file): {filename}")
            continue

        try:
            question_id = int(filename.lower().split("question_")[1].split(".")[0])
        except:
            print(f"⚠️ Tidak bisa membaca question_id dari: {filename}")
            continue

        # Temp copy
        temp_video = copy_to_temp(video_path)

        try:
            output = process_single_video(temp_video, question_id, payload, idx)
            results.append(output)

        finally:
            cleanup_audio(temp_video)
            if os.path.exists(temp_video):
                os.remove(temp_video)

    return results


# =============================
# ENTRYPOINT
# =============================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_batch_evaluator.py <folder_path>\n")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.exists(folder):
        print(f"❌ Folder tidak ditemukan: {folder}")
        sys.exit(1)

    payload = load_payload()

    print("🚀 Menjalankan batch evaluator...\n")
    final_results = batch_evaluate(folder, payload)

    print(json.dumps(final_results, indent=2, ensure_ascii=False))
    print("\n Selesai!\n")
