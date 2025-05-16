from nltk.translate.bleu_score import sentence_bleu

reference = ["Привет, ты стал больше", "В разы больше ты стал"]
candidate = "Привет, ты стал в разы больше"

score = sentence_bleu(reference, candidate)
print(f"BLEU-4: {score:.4f}")