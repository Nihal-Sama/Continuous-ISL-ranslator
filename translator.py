class SmartTranslator:
    def __init__(self):
        print("✅ Lightweight Translator Initialized (Memory Saver Mode)")

    def enhance_sentence(self, glosses):
        if not glosses: return ""
        
        # Join the detected signs
        raw_text = " ".join(glosses).lower().strip()
        
        # Basic smart formatting to look professional for the presentation
        if raw_text:
            # Capitalize first letter and add a period
            final_text = raw_text.capitalize() + "."
            return final_text
            
        return raw_text
