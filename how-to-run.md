
# Setup Guide

## 1. Clone Repository

Silakan **fork** terlebih dahulu (opsional), lalu clone repo berikut:

```bash
git clone https://github.com/doctordoom101/ai-interview-assesment.git
cd ai-interview-assesment
```

---

## 2. Download Whisper Large-v2 Model

1. Buka link Google Drive berikut:
   [https://drive.google.com/drive/folders/1S0MsXMc5k4CexZ-fmmd6V_ZF4OhaayXp?usp=sharing](https://drive.google.com/drive/folders/1S0MsXMc5k4CexZ-fmmd6V_ZF4OhaayXp?usp=sharing)

2. Download model **whisper-large-v2-en.zip**

3. Extract, lalu tempatkan di project directory:

```
├── models/
│   └── whisper-large-v2-en/
│       ├── added_tokens.json
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── model.safetensors.index.json
│       ├── normalizer.json
│       ├── preprocessor_config.json
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.json
```

---

## 3. Build Containers

```bash
docker compose build
```

---

## 4. Start All Services

```bash
docker compose up -d
```

### UI (Streamlit)

Buka browser:
**[http://localhost:8501](http://localhost:8501)**

### API Docs (FastAPI)

Buka:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 5. Check Containers

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f streamlit
```

---

## 6. Pull Ollama Model (Optional, lighter model)

```bash
docker compose exec ollama ollama pull llama3.2
```

---

## 7. Run Evaluator (Single File)

```bash
docker compose run --rm api python test_evaluator1.py assets/videos/interview_question_1.webm 1
```

Atau:

```bash
docker compose exec api python test_evaluator1.py assets/videos/interview_question_1.webm 1
```

---

## 8. Run Batch Evaluator

```bash
docker compose run --rm api python test_batch_evaluator.py assets/videos
```

Atau:

```bash
docker compose exec api python test_batch_evaluator.py assets/videos
```

---

## 9. Stop All Containers

```bash
docker compose down
```

---

## 10. Full Cleanup

```bash
docker compose down --volumes --rmi all
```

