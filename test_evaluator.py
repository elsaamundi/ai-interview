from utils.transcript_evaluator import evaluate_transcript

# Contoh transkrip dummy untuk tes awal
transcript_text = """
Can you share any specific challenges you face while working on certification? How you overcome them? Ah, okay, actually, for the challenges, there are some challenges when I took the certifications, especially for the projects I mentioned that I already working with it. The first one is actually to meet the specific accuracy or validation loss for the evaluation metrics. And yeah, actually, that's just need to take some trial and error with different architecture. For example, like we can try to add more layer, more neurons, changes the neurons, or even I also apply the dropout layer. So yeah, it really helps with the foundation loss to become more lower, right? And yeah, I think that's one of the biggest challenges that I have while working on these certifications.
"""

# Contoh pertanyaan (Question 1)
question_id = 1
question = "Can you share any specific challenges you faced while working on certification and how you overcame them?"

# Jalankan evaluator
result = evaluate_transcript(
    question_id=question_id,
    question=question,
    answer=transcript_text
)

print("=== Evaluator Output ===")
print(result)
