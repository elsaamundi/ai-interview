from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import os
import json

from utils.speech_to_text import transcribe_video
from utils.eye_focus_detection import process_video_for_gaze
from utils.transcript_evaluator import evaluate_transcript


app = FastAPI(title="AI Interview Backend API")


# ======================================================
# Helpers
# ======================================================
def load_payload(path="data/payload.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================
# API: Single Processing
# ======================================================
@app.post("/process/single")
async def process_single(
    file: UploadFile = File(...),
    enable_evaluator: bool = Form(True)
):
    # -----------------------------
    # Save temp video
    # -----------------------------
    suffix = "." + file.filename.split(".")[-1]
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(await file.read())
    temp.flush()
    temp.close()

    video_path = temp.name

    # -----------------------------
    # 1. Transcription
    # -----------------------------
    transcript_text = transcribe_video(
        video_path,
        prompt="This audio is an English HR interview. Transcribe clearly."
    )

    # -----------------------------
    # 2. Eye Focus
    # -----------------------------
    try:
        gaze_result = process_video_for_gaze(video_path)
    except Exception as e:
        gaze_result = {"status": "failed", "error": str(e)}

    # -----------------------------
    # 3. Evaluator (optional)
    # -----------------------------
    evaluation = None
    if enable_evaluator:

        # extract question id
        base = os.path.basename(file.filename)
        try:
            question_id = int(base.split("_")[-1].split(".")[0])
        except:
            os.remove(video_path)
            return JSONResponse(
                status_code=400,
                content={"error": "Filename must contain question ID, e.g. video_12.mp4"}
            )

        payload = load_payload()
        items = payload["data"]["reviewChecklists"]["interviews"]

        item = next((q for q in items if q["positionId"] == question_id), None)
        if not item:
            os.remove(video_path)
            return JSONResponse(
                status_code=400,
                content={"error": f"Question ID {question_id} not found in payload"}
            )

        question_text = item["question"]

        evaluation = evaluate_transcript(
            question_id=question_id,
            question=question_text,
            answer=transcript_text
        )

    # cleanup
    os.remove(video_path)

    return {
        "transcription": transcript_text,
        "evaluation": evaluation,
        "eye_focus": gaze_result
    }


# ======================================================
# API: Batch Processing (folder inside container)
# ======================================================
@app.post("/process/batch")
def process_batch(folder_path: str = Form(...)):

    if not os.path.exists(folder_path):
        return {"error": "Folder not found"}

    supported_ext = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    payload = load_payload()

    results = []

    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext not in supported_ext:
            continue

        full_path = os.path.join(folder_path, f)

        # extract question id
        try:
            qid = int(f.lower().split("question_")[1].split(".")[0])
        except:
            continue

        # temp copy
        suffix = "." + full_path.split(".")[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with open(full_path, "rb") as src:
            tmp.write(src.read())
        tmp.flush()
        tmp.close()

        # ---- main process ----
        transcript = transcribe_video(tmp.name)
        try:
            gaze = process_video_for_gaze(tmp.name)
        except:
            gaze = None

        items = payload["data"]["reviewChecklists"]["interviews"]
        item = next((q for q in items if q["positionId"] == qid), None)

        if item:
            eval_result = evaluate_transcript(
                question_id=qid,
                question=item["question"],
                answer=transcript
            )
        else:
            eval_result = None

        results.append({
            "file": f,
            "transcript": transcript,
            "evaluation": eval_result,
            "eye_focus": gaze
        })

        os.remove(tmp.name)

    return {"results": results}
