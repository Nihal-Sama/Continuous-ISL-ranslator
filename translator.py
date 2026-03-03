from transformers import pipeline

class SmartTranslator:
    def __init__(self):
        print("Attempting to load NLP Model (T5)...")
        self.pipe = None
        try:
            # Auto-detects task, preventing KeyErrors
            self.pipe = pipeline(model="t5-small")
            print("✅ NLP Model Loaded Successfully!")
        except Exception as e:
            print(f"⚠️ NLP Loading Warning: {e}")
            print("➡️ Switching to BASIC MODE (No Grammar Correction).")
            self.pipe = None

    def enhance_sentence(self, glosses):
        if not glosses: return ""
        
        raw_text = " ".join(glosses).lower()
        if self.pipe is None:
            return raw_text
            
        try:
            prompt = f"convert to English sentence: {raw_text}"
            result = self.pipe(prompt, max_length=50, num_beams=5, early_stopping=True)
            return result[0]['generated_text']
        except Exception:
            return raw_text