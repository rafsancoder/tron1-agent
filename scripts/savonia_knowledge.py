"""Verify a narrow Savonia request against a small, local, sourced fact file."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from scripts.entity_resolution import candidate_evidence, compact


CLARIFY = {
    "en": "Do you mean Savonia University of Applied Sciences in Kuopio?",
    "fi": "Tarkoitatko Savonia-ammattikorkeakoulua Kuopiossa?",
}
MISSING = {
    "en": "The local Savonia knowledge file does not contain that information. Please check Savonia's official website.",
    "fi": "Paikallinen Savonia-tietopohja ei sisällä tätä tietoa. Tarkista tieto Savonian virallisilta verkkosivuilta.",
}
OUTSIDE = {
    "en": "The local file covers Savonia. I have no verified information about that institution.",
    "fi": "Paikallinen tietopohja käsittelee Savoniaa. Minulla ei ole vahvistettua tietoa kyseisestä korkeakoulusta.",
}
ASK_TOPIC = {
    "en": "Which place or institution do you mean?",
    "fi": "Mitä paikkaa tai korkeakoulua tarkoitat?",
}
REJECTED = {"en": "Which institution do you mean?", "fi": "Mitä korkeakoulua tarkoitat?"}
INSTITUTION = re.compile(r"\bsavonia(?:n|ssa|sta|an|lla|lta|lle)?\b")
SPELLED_NAME = re.compile(r"\b" + r"[\s\-–—]+".join("savonia") + r"\b")
SPACED_NAME = re.compile(r"\bsavo[\s\-–—]+nia\b")
UNCERTAIN = re.compile(r"\b(?:savona|savonie|savionia)(?:n|ssa|sta|an|lla|lle)?\b")
PHONETIC_NAME = re.compile(r"\b(?:sabuna|sabunya|saboenia|saboven|sabo[\s-]+nia)\b")
NAMED_SCHOOL = re.compile(r"\b([a-zäöå][\w-]*)\s+(?:universit(?:y|ies)|uas|yliopisto\w*|ammattikorkeakoulu\w*|amk)\b")
SCHOOL_OF = re.compile(r"\b(?:university|college)\s+of\s+(?!applied\b)([a-zäöå][\w-]*)\b")
OTHER_SCHOOL = re.compile(
    r"\b(?:university of (?:helsinki|eastern finland|jyväskylä|turku|oulu|tampere)|"
    r"(?:aalto|tampere) university|helsingin yliopisto|itä-suomen yliopisto|"
    r"jyväskylän yliopisto|aalto-yliopisto|metropolia(?: university of applied sciences| ammattikorkeakoulu)?)\b"
)
APPLIED_SCIENCES = re.compile(r"\b(?:university of applied sciences?|ammattikorkeakoulu\w*)\b")
LOCATION_WORD = re.compile(r"\b(?:located|location|where (?:is|are)|missä|sijaitse\w*)\b")
OVERVIEW = re.compile(r"\b(?:know about|tell me about|kerro)\b")
NON_NAMES = {"a", "an", "the", "this", "that", "which", "what", "your", "our", "local", "nearest", "is", "at", "of",
             "in", "to", "mikä", "se", "tämä", "lähin"}
INSTITUTION_MENTION = re.compile(
    r"(?:savonia(?:[- ]ammattikorkeakoulu)?|sabuna|sabunya|saboenia|saboven|sabo[\s-]+nia|savo[\s-]+nia)"
    r"(?:\s+(?:uas|university(?:\s+of\s+applied\s+sciences?)?))?"
    r"(?:[\s.,]+(?:coopio|kuopio|finland))*[\s.?!]*"
)
CONFIRM = re.compile(r"(?:yes(?:\s+please)?|kyllä|joo)(?:\s*,?\s+(?:i mean (?:it|that)|tarkoitan sitä|juuri sitä))?")
REJECT = re.compile(r"(?:no|nope|ei|en)(?:\s*,?\s+(?:not that|i mean something else|en tarkoita sitä|tarkoitan muuta))?")
PLACE = re.compile(r"\b(?:kuopio|iisalm[ie]|varkau[sd])\w*\b")
SCHOOL = re.compile(r"\b(?:universit\w*|campus\w*|ammattikorkeakoulu\w*|yliopisto\w*|kampu\w*|amk)\b")
EDUCATION = re.compile(r"\b(?:study|studies|degree\w*|opiskel\w*|koulutu\w*|tutkin\w*)\b")
FOLLOW_UP = re.compile(r"\b(?:it|its|they|their|there|the university|se|sen|siellä|niiden|entä)\b")
IMPLICIT_LOCATION = re.compile(r"\b(?:where (?:is|are) (?:it|they|located)|missä (?:se|ne) (?:on|ovat|sijaitse\w*))\b")
NAME = re.compile(r"\b(?:official name|finnish name|english name|virallinen nimi|"
                  r"nimi suomeksi|nimi englanniksi|suomenkielinen nimi|englanninkielinen nimi)\b")
UNVERIFIED = re.compile(
    r"\b(?:hours?|opens?|opening|today|tomorrow|current|room\w*|staff|schedule\w*|deadline\w*|fees?|tuition|directions|café|cafeteria|"
    r"found(?:er\w*|ed)|ranking\w*|ranked|how many|kuinka monta|montako|perusta\w*|sijoitus\w*|"
    r"auki\w*|tänään|huomenna|huone\w*|henkilökun\w*|aikataulu\w*|lukujärjestys\w*|maksu\w*|reitti\w*|kahvil\w*)\b"
    r"|how (?:do|can) i get|miten pääsen"
)
TOPICS = {
    "international": r"\b(?:international\w*|exchange\w*|english.taught|in english|kansainväl\w*|vaihto\w*|englanniksi|englanninkiel\w*)\b",
    "studies": r"\b(?:stud(?:y|ies)|degree\w*|program\w*|educat\w*|opiskel\w*|opinno\w*|koulutu\w*|tutkin\w*)\b",
    "campuses": r"\b(?:campus\w*|where is (?:savonia|sabuna|sabunya|saboenia|saboven|sabo[\s-]+nia)|kampu\w*|"
                r"missä savonia on|missä savonia sijaitsee|paikkakunn\w*)\b",
    "contact": r"\b(?:contact\w*|website\w*|links?|application\w*|admission\w*|yhteys\w*|verkkosivu\w*|link\w*|hakemi\w*)\b",
    "identity": r"\b(?:what is savonia(?!['’]|\s+\w)|what kind of (?:institution|school)|tell me about savonia|"
                r"official name|mikä savonia on|millainen korkeakoulu|kerro savoniasta|virallinen nimi)\b",
}


@dataclass(frozen=True)
class Interpretation:
    entity: str
    evidence: tuple[str, ...]
    original_question: str


@dataclass
class KnowledgeState:
    active: Interpretation | None = None
    pending: Interpretation | None = None
    person: str | None = None
    person_pending: tuple[Interpretation, ...] = ()


@dataclass(frozen=True)
class KnowledgeContext:
    reply: str
    route: str
    references: tuple[tuple[str, str], ...] = ()
    interpretation: Interpretation | None = None
    fact_ids: tuple[str, ...] = ()
    provenance: str | None = None
    candidates: tuple[Interpretation, ...] = ()


def school_spans(query):
    after = SCHOOL_OF.search(query)
    if after:
        return (after[1],)
    for match in NAMED_SCHOOL.finditer(query):
        if match[1] not in NON_NAMES:
            before = re.findall(r"[a-zäöå][\w-]*", query[:match.end(1)])
            return tuple(" ".join(before[-length:]) for length in range(1, min(len(before), 3) + 1))
    return ()


def institution_candidate(query, spans, recent):
    school = SCHOOL.search(query)
    study = EDUCATION.search(query)
    signal = school or study
    alias = (signal and UNCERTAIN.search(query)) or (school and PHONETIC_NAME.search(query))
    if alias:
        return (f"documented similar name: {alias[0]}", f"institution or study term: {signal[0]}")
    if not school:
        return ()
    context = [f"institution term: {school[0]}"]
    if applied := APPLIED_SCIENCES.search(query):
        context.append(f"institution type: {applied[0]}")
    if recent:
        context.append("recent confirmed Savonia topic")
    if place := PLACE.search(query):
        context.append(f"known campus city: {place[0]}")
    else:
        for span in re.findall(r"\b[a-zäöå]+(?:-[a-zäöå]+)*\b", query):
            if not 5 <= len(compact(span)) <= 8:
                continue
            if candidate_evidence(span, ("Kuopio",), ("institution context",), spelling_min=0.85, sound_min=1):
                context.append(f"possible Kuopio mention: {span}")
                break
    contextual = len(context) > 1
    for span in spans:
        if evidence := candidate_evidence(span, ("Savonia",), context,
                                           spelling_min=0.80 if contextual else 0.88,
                                           sound_min=0.85 if contextual else 1):
            return evidence
    return tuple(context) if not spans and place else ()


class SavoniaKnowledge:
    def __init__(self, *, profile=None):
        from scripts.savonia_people import SavoniaPeople

        self.path = Path(__file__).resolve().with_name("savonia_facts.json")
        self.facts = json.loads(self.path.read_text(encoding="utf-8"))
        self.people = SavoniaPeople(profile)

    def _answer(self, lang, topic, interpretation):
        fact = self.facts[topic or "contact"]
        references = tuple((url, fact["verified_on"]) for url in fact["sources"])
        if topic is None:
            return KnowledgeContext(MISSING[lang], "missing fact", references, interpretation)
        return KnowledgeContext(fact[lang], f"matched topic: {topic}", references, interpretation, (topic,))

    def _topic(self, query):
        if UNVERIFIED.search(query):
            return None
        if NAME.search(query):
            return "identity"
        if IMPLICIT_LOCATION.search(query):
            return "campuses"
        for topic, pattern in TOPICS.items():
            if re.search(pattern, query):
                return topic
        if SPELLED_NAME.search(query) or SPACED_NAME.search(query) or INSTITUTION_MENTION.fullmatch(query):
            return "identity"
        return None

    def lookup(self, question, lang, state=None):
        # Lookup normalization never modifies the stored question.
        state = state if state is not None else KnowledgeState()
        query = question.strip().casefold()
        if personal := self.people.lookup(question, lang, state):
            state.active = state.pending = None
            return personal
        pending, state.pending = state.pending, None
        if pending and REJECT.fullmatch(query.rstrip(".!?")):
            state.active = None
            return KnowledgeContext(REJECTED[lang], "confirmation rejected", interpretation=pending)
        if pending and CONFIRM.fullmatch(query.rstrip(".!?")):
            state.active = Interpretation(pending.entity, (*pending.evidence, "user confirmation"), pending.original_question)
            original = pending.original_question.strip().casefold()
            topic = self._topic(original)
            if topic is None and not UNVERIFIED.search(original):
                if LOCATION_WORD.search(original):
                    topic = "campuses"
                elif OVERVIEW.search(original):
                    topic = "identity"
            return self._answer(lang, topic, state.active)

        exact = INSTITUTION.search(query)
        letters = SPELLED_NAME.search(query) or SPACED_NAME.search(query)
        if exact or letters:
            evidence = f"explicit spelling: {letters[0]}" if letters else f"exact name: {exact[0]}"
            state.active = Interpretation("Savonia", (evidence,), question)
            return self._answer(lang, self._topic(query), state.active)

        if other := OTHER_SCHOOL.search(query):
            state.active = Interpretation(other[0], ("explicit different institution name",), question)
            return KnowledgeContext(OUTSIDE[lang], "missing fact: other institution", interpretation=state.active)
        spans = school_spans(query)
        recent = state.active is not None and state.active.entity == "Savonia"
        if evidence := institution_candidate(query, spans, recent):
            state.pending = Interpretation("Savonia", evidence, question)
            state.active = None
            return KnowledgeContext(CLARIFY[lang], "clarification", interpretation=state.pending)
        if spans:
            state.active = None
            return KnowledgeContext(REJECTED[lang], "clarification: unknown institution")

        topic = self._topic(query)
        follow_up = IMPLICIT_LOCATION.search(query) or (FOLLOW_UP.search(query) and (topic or UNVERIFIED.search(query)))
        if follow_up:
            if state.active is None:
                return KnowledgeContext(ASK_TOPIC[lang], "clarification: ambiguous referent")
            if state.active.entity != "Savonia":
                return KnowledgeContext(OUTSIDE[lang], "missing fact: other institution", interpretation=state.active)
            interpretation = Interpretation("Savonia", (*state.active.evidence, "immediate topic follow-up"), question)
            return self._answer(lang, topic, interpretation)

        state.active = None
        return None
