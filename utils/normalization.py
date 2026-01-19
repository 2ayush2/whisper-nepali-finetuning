import re
import unicodedata

ZERO_WIDTH_CHARS = (
    "\u200b"  
    "\u200c"  
    "\u200d"  
    "\ufeff"  
)
def normalize_nepali_text(text: str) -> str:
    """
    Normalizes Nepali text for Whisper ASR.
    - Uses NFC (Canonical) normalization.
    - Preserves punctuation and numbers.
    - Removes zero-width/invisible characters.
    - Normalizes whitespace.
    """
    if not text:
        return ""
    # 1. Canonical Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # 2. Remove invisible/zero-width characters
    text = re.sub(f"[{ZERO_WIDTH_CHARS}]", "", text)
    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # 4. Fix danda spacing (optional but recommended)
    text = re.sub(r"\s+।", "।", text)
    return text
