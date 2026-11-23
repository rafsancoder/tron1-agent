"""Bounded comparisons of names extracted by the local knowledge router."""

from difflib import SequenceMatcher
import re
import unicodedata


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold()) if not unicodedata.combining(c))


def compact(text):
    return re.sub(r"[^a-z]", "", fold(text))


def sound_key(text):
    name = compact(text).replace("gh", "f").replace("bh", "v")
    consonants = re.sub(r"[aeiouy]", "", name).translate(str.maketrans("dgbc", "tkvk"))
    consonants = consonants.replace("vh", "v")
    return re.sub(r"(.)\1+", r"\1", consonants)


def candidate_evidence(span, forms, context, *, spelling_min=0.72, sound_min=0.80):
    """Compare only extracted spans and only when routing context supports it."""
    name = compact(span)
    if not context or len(name) < 4:
        return ()
    forms = tuple(compact(form) for form in forms)
    spelling = max(SequenceMatcher(None, name, form).ratio() for form in forms)
    key = sound_key(name)
    sound = max(SequenceMatcher(None, key, sound_key(form)).ratio() for form in forms) if len(key) >= 2 else 0
    if spelling < spelling_min and sound < sound_min:
        return ()
    return (*context, f"extracted name: {span}", f"spelling similarity: {spelling:.2f}",
            f"sound-key similarity: {sound:.2f}")
