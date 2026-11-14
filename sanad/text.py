"""Arabic text normalisation and tokenisation for lexical retrieval."""
from __future__ import annotations

import re
import unicodedata

DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ALEF = re.compile(r"[آأإٱ]")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
WORD = re.compile(r"[\w]+", re.UNICODE)
PREFIXES = ("وبال", "وكال", "ولل", "وال", "بال", "كال", "فال", "لل", "ال")
STOP = set("""في من على الى إلى عن او أو ان أن إن ما لا لم لن هذا هذه ذلك تلك التي الذي الذين و ف ب ل ك ثم قد كل
مع عند بين هو هي هم كان كانت يكون تكون اي أي غير بعد قبل حتى اذا إذا""".split())


def normalise(text: str) -> str:
    """Spelling variants that should compare equal: diacritics, tatweel, hamza-on-alef, alef maqsura, taa marbuta, digits."""
    text = unicodedata.normalize("NFKC", text)
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = ALEF.sub("ا", text).replace("ى", "ي").replace("ة", "ه")
    return text.translate(DIGITS).lower()


def light_stem(word: str) -> str:
    for p in PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 3:
            return word[len(p):]
    if word.startswith("و") and len(word) >= 5:
        return word[1:]
    return word


def tokens(text: str) -> list[str]:
    out = []
    for w in WORD.findall(normalise(text)):
        if w in STOP:
            continue
        out.append(light_stem(w))
    return out


def key(text: str) -> str:
    """Whitespace- and punctuation-insensitive form used to find the same legal text in two datasets."""
    return " ".join(WORD.findall(normalise(text)))
