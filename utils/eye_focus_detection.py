import cv2
import mediapipe as mp
import numpy as np
import os
import time

def get_gaze_direction(face_landmarks):
    """
    Menganalisis landmark wajah untuk menentukan arah pandang horizontal.
    Menggunakan rasio posisi iris relatif terhadap sudut mata.
    
    Args:
        face_landmarks: Objek landmark wajah dari MediaPipe.
        
    Returns:
        str: "Kiri", "Kanan", "Tengah", atau "Tidak Terdeteksi".
    """
    try:
        # Landmark Mata Kiri (Indeks Landmark MediaPipe)
        # Sudut kiri: 33, Sudut kanan: 133, Iris: 473
        left_corner_x = face_landmarks.landmark[33].x
        right_corner_x = face_landmarks.landmark[133].x
        iris_center_x = face_landmarks.landmark[473].x 

        #Landmark Mata Kanan
        # Sudut kiri: 362, Sudut kanan: 263, Iris: 468
        left_corner_x_right_eye = face_landmarks.landmark[362].x
        right_corner_x_right_eye = face_landmarks.landmark[263].x
        iris_center_x_right_eye = face_landmarks.landmark[468].x 
        
        # Mata Kiri
        left_eye_width = right_corner_x - left_corner_x
        # Hindari pembagian dengan nol (jika mata tertutup/tidak terdeteksi)
        if left_eye_width > 0.001:
            left_eye_ratio = (iris_center_x - left_corner_x) / left_eye_width
        else:
            left_eye_ratio = 0.5 

        # Mata Kanan
        right_eye_width = right_corner_x_right_eye - left_corner_x_right_eye
        if right_eye_width > 0.001:
            right_eye_ratio = (iris_center_x_right_eye - left_corner_x_right_eye) / right_eye_width
        else:
            right_eye_ratio = 0.5

        # Ambil rata-rata rasio kedua mata untuk stabilitas
        gaze_ratio = (left_eye_ratio + right_eye_ratio) / 2


        if gaze_ratio < 0.45: 
            return "Kanan" 
        elif gaze_ratio > 0.58: 
            return "Kiri" 
        else:
            return "Tengah"
    
    except Exception:
        return "Tidak Terdeteksi"


def analyze_gaze_log(log, fps):
    """
    Mengubah data mentah (log per frame) menjadi laporan statistik JSON.
    
    Args:
        log (list): Daftar arah pandang per frame ["Tengah", "Kiri", ...].
        fps (float): Frame per second dari video.
        
    Returns:
        dict: Laporan analisis lengkap.
    """
    total_frames = len(log)
    if total_frames == 0:
        return {"status": "failed", "error": "Tidak ada frame yang berhasil diproses."}
    
    # Gunakan default FPS jika gagal dideteksi
    if fps is None or fps <= 0 or fps > 100:
        print(f"Peringatan: FPS terdeteksi {fps} (tidak wajar). Menggunakan default 30 FPS.")
        fps = 30.0 

    # Filter frame yang valid (wajah terdeteksi)
    valid_frames = [x for x in log if x != "Wajah Tidak Terdeteksi"]
    total_detected = len(valid_frames)
    
    if total_detected == 0:
         return {"status": "failed", "error": "Wajah tidak terdeteksi sama sekali dalam video."}

    # Hitung Statistik Dasar
    tengah_count = valid_frames.count("Tengah")
    kiri_count = valid_frames.count("Kiri")
    kanan_count = valid_frames.count("Kanan")
    
    focus_percentage = (tengah_count / total_detected) * 100
    left_percentage = (kiri_count / total_detected) * 100
    right_percentage = (kanan_count / total_detected) * 100

    # Deteksi Indikator Kecurangan (Suspicious Events)
    # Logika: Melirik ke samping secara terus-menerus selama > 2 detik
    suspicious_events = []
    consecutive_threshold_frames = int(fps * 2)
    
    i = 0
    while i < total_frames:
        direction = log[i]
        # Jika sedang melihat ke samping (Kiri atau Kanan)
        if direction == "Kiri" or direction == "Kanan":
            start_frame = i
            # Hitung berapa lama (frame) dia bertahan melihat ke sana
            while i + 1 < total_frames and log[i+1] == direction:
                i += 1
            
            consecutive_frames = (i - start_frame) + 1
            duration_seconds = consecutive_frames / fps
            
            # Jika durasinya melebihi threshold (2 detik), catat sebagai mencurigakan
            if consecutive_frames >= consecutive_threshold_frames:
                suspicious_events.append({
                    "start_time_seconds": round(start_frame / fps, 2),
                    "end_time_seconds": round(i / fps, 2),
                    "duration_seconds": round(duration_seconds, 2),
                    "direction": direction
                })
        i += 1

    # Buat Kesimpulan Otomatis (Summary Note)
    summary_note_cv = ""
    suspicious_count = len(suspicious_events)

    if focus_percentage > 85:
        summary_note_cv = "Kandidat sangat fokus ke kamera (fokus > 85%). Perilaku tidak mencurigakan."
    elif focus_percentage < 50:
        summary_note_cv = "PERINGATAN: Tingkat fokus kandidat sangat rendah (< 50%). Indikasi kuat ketidakwajaran."
    elif suspicious_count > 0:
        summary_note_cv = f"Kandidat cukup fokus, namun terdeteksi {suspicious_count} kali mengalihkan pandangan cukup lama (>2 detik)."
    else:
        summary_note_cv = f"Kandidat cukup fokus ke kamera ({focus_percentage:.0f}%). Tidak ada indikasi mencurigakan yang signifikan."

    # Susun Laporan Akhir (Kontrak Data)
    return {
        "status": "success",
        "video_duration_seconds": round(total_frames / fps, 2),
        "analysis_fps": round(fps, 2),
        "focus_percentage": round(focus_percentage, 2),
        "left_glance_percentage": round(left_percentage, 2),
        "right_glance_percentage": round(right_percentage, 2),
        "suspicious_event_count": suspicious_count,
        "suspicious_events_list": suspicious_events,
        "summary_note_cv": summary_note_cv
    }

def process_video_for_gaze(video_path):
    """
    Fungsi utama untuk memproses video dari awal sampai akhir.
    
    Args:
        video_path (str): Path file video input.
        
    Returns:
        dict: Laporan hasil analisis (JSON compatible).
    """
    # Validasi Input
    if not os.path.exists(video_path):
        return {"status": "failed", "error": f"File tidak ditemukan: {video_path}"}

    # Inisialisasi MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    try:
        # Gunakan 'with' statement untuk manajemen memori yang aman
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,       
            max_num_faces=1,               
            refine_landmarks=True,         
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"status": "failed", "error": "Gagal membuka file video dengan OpenCV"}

            fps = cap.get(cv2.CAP_PROP_FPS)
            gaze_log = []

            # Loop per frame
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Konversi warna BGR (OpenCV) ke RGB (MediaPipe)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Agar performa lebih cepat, tandai image sebagai tidak bisa diubah (not writeable)
                image_rgb.flags.writeable = False
                
                # Deteksi Wajah
                results = face_mesh.process(image_rgb)

                # Analisis Hasil Deteksi
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    direction = get_gaze_direction(face_landmarks)
                    gaze_log.append(direction)
                else:
                    gaze_log.append("Wajah Tidak Terdeteksi")

            # Bersihkan resource video
            cap.release()
            
            # Lakukan analisis statistik pada data yang terkumpul
            final_report = analyze_gaze_log(gaze_log, fps)
            
            return final_report

    except Exception as e:
        return {"status": "failed", "error": f"Terjadi kesalahan sistem: {str(e)}"}

if __name__ == "__main__":
    # Ganti dengan path video lokal untuk tes
    test_video = os.path.join(os.path.dirname(__file__), "..", "assets", "videos", "interview_question_1.webm")
    test_video = os.path.abspath(test_video)
    
    if os.path.exists(test_video):
        print(f"Sedang menguji modul dengan video: {test_video}...")
        start_time = time.time()
        
        report = process_video_for_gaze(test_video)
        
        print(f"Waktu proses: {time.time() - start_time:.2f} detik ")
        print("\n=== HASIL LAPORAN ===")
        # Gunakan json.dumps untuk mencetak dictionary dengan rapi (pretty print)
        import json
        print(json.dumps(report, indent=4))
    else:
        print("Info: File ini adalah modul library. Untuk menguji, sediakan file 'video_sample.mp4'.")