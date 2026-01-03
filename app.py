import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://capstone_api:8000")

st.set_page_config(page_title="AI Interview UI", layout="wide")

st.sidebar.title("Menu")
mode = st.sidebar.radio("Mode:", ["Single Processing", "Batch Processing"])


# ======================================================
# MODE 1 — Single Processing
# ======================================================
if mode == "Single Processing":
    st.title("🎙️ Single Video Processing")

    video_file = st.file_uploader(
        "Upload video interview",
        type=["mp4", "mov", "mkv", "avi", "webm"]
    )

    enable_eval = st.checkbox("Enable Evaluator", value=True)

    if st.button("Execute"):
        if not video_file:
            st.error("Please upload a video.")
            st.stop()

        with st.spinner("Processing..."):
            res = requests.post(
                f"{API_URL}/process/single",
                files={"file": (video_file.name, video_file.getvalue())},
                data={"enable_evaluator": str(enable_eval)}
            )

        if res.status_code != 200:
            st.error(res.text)
        else:
            data = res.json()

            # ==========================
            # TRANSCRIPT SECTION
            # ==========================
            st.markdown("## 📝 Transcription")
            st.text_area("", data["transcription"], height=250)

            # ==========================
            # EYE FOCUS SECTION
            # ==========================
            focus = data["eye_focus"]

            st.markdown("## 👁️ Eye Focus Analysis")

            colA, colB = st.columns(2)

            with colA:
                st.metric("Focus %", f"{focus['focus_percentage']:.2f}%")
                st.metric("Left Glance", f"{focus['left_glance_percentage']:.2f}%")
                st.metric("Right Glance", f"{focus['right_glance_percentage']:.2f}%")

            with colB:
                st.metric("Duration (s)", f"{focus['video_duration_seconds']:.2f}")
                st.metric("FPS", f"{focus['analysis_fps']:.2f}")
                st.metric("Suspicious Events", focus["suspicious_event_count"])

            st.progress(focus["focus_percentage"] / 100)

            if focus["suspicious_event_count"] > 0:
                st.markdown("### ⚠️ Suspicious Events")
                st.table(focus["suspicious_events_list"])
            else:
                st.success("No suspicious events detected.")

            st.info(f"📌 Summary: {focus['summary_note_cv']}")

            # ==========================
            # EVALUATION SECTION
            # ==========================
            if enable_eval and "evaluation" in data:
                eval_data = data["evaluation"]
                st.markdown("## 🧮 Evaluation Result")

                # Score badge style
                score_color = (
                    "green" if eval_data["score"] >= 4 else
                    "orange" if eval_data["score"] == 3 else
                    "red"
                )

                st.markdown(
                    f"""
                    <div style="
                        background-color:{score_color};
                        color:white;
                        padding:10px;
                        border-radius:8px;
                        width:110px;
                        text-align:center;
                        font-size:22px;">
                        Score: {eval_data['score']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### 🔍 Reasoning")
                st.write(eval_data["reason"])

# ======================================================
# MODE 2 — Batch Processing
# ======================================================
if mode == "Batch Processing":
    st.title("📦 Batch Video Processing")

    folder_path = st.text_input("Folder path inside container:")

    if st.button("Run Batch"):
        if not folder_path:
            st.error("Folder path required.")
            st.stop()

        with st.spinner("Processing batch..."):
            res = requests.post(
                f"{API_URL}/process/batch",
                data={"folder_path": folder_path}
            )

        if res.status_code != 200:
            st.error(res.text)
        else:
            data = res.json()
            results = data.get("results", [])

            st.subheader("📊 Batch Processing Results")

            # Summary Table (file, score, focus %, suspicious count)
            st.write("### Summary Table")

            summary_rows = []
            for item in results:
                summary_rows.append({
                    "File": item["file"],
                    "Score": item["evaluation"]["score"],
                    "Focus (%)": item["eye_focus"]["focus_percentage"],
                    "Suspicious Events": item["eye_focus"]["suspicious_event_count"]
                })

            st.dataframe(summary_rows)

            # Detailed per row
            st.write("### Detailed Breakdown")

            for item in results:
                with st.expander(f"📁 {item['file']}"):
                    st.write("#### 📝 Transcript")
                    st.write(item["transcript"])

                    st.write("#### 🧠 Evaluation")
                    st.metric("Score", item["evaluation"]["score"])
                    st.write("Reason:", item["evaluation"]["reason"])

                    st.write("#### 👁️ Eye Focus Analysis")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Focus %", f"{item['eye_focus']['focus_percentage']:.2f}%")
                    col2.metric("Left Glance %", f"{item['eye_focus']['left_glance_percentage']:.2f}%")
                    col3.metric("Right Glance %", f"{item['eye_focus']['right_glance_percentage']:.2f}%")

                    st.write("Suspicious Events:", item["eye_focus"]["suspicious_event_count"])
                    if item["eye_focus"]["suspicious_events_list"]:
                        st.write(item["eye_focus"]["suspicious_events_list"])
                    else:
                        st.write("No suspicious events.")

                    st.write("Summary:", item["eye_focus"]["summary_note_cv"])

