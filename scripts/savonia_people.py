"""Select local profile answers and verified staff facts without an LLM lookup."""

from dataclasses import replace
import json
from pathlib import Path
import re

from scripts.entity_resolution import candidate_evidence, compact, fold
from scripts.savonia_knowledge import CONFIRM, REJECT, Interpretation, KnowledgeContext


PROFILE_DIR = Path(__file__).resolve().parents[1] / "asset" / "personal-profiles"
EMAIL = re.compile(r"\b[a-z0-9._+-]+@savonia\.fi\b", re.IGNORECASE)
PERSONAL = {
    "full_name": r"\b(?:what (?:is|was) my (?:full )?name|what's my (?:full )?name|who am i|kuka minä olen|mikä on (?:koko)?nimeni)\b",
    "supervisor": r"\b(?:my .{0,24}supervisor|who supervises my|ohjaajani|(?:opinnäytetyöni|harjoitteluni) ohjaaja)\b",
    "internship": r"\b(?:my internship|what work am i doing|what do i work on|harjoitteluni|mitä työtä teen|robot savo)\b",
    "thesis": r"\b(?:my thesis|opinnäytetyöni|mikä on opinnäytetyöni aihe)\b",
    "education": r"\b(?:my education|what (?:am i studying|do i study)|what is my degree|mitä opiskelen|koulutukseni|tutkintoni)\b",
    "interests": r"\b(?:my interests|what am i interested in|mistä olen kiinnostunut|kiinnostuksenkohteeni)\b",
}
TEACHER = re.compile(r"\b(?:my teacher|opettajani)\b")
EMAIL_QUERY = re.compile(r"\b(?:e-?mail\w*|sähköposti\w*|työsähköposti\w*)\b")
REFERENT = re.compile(r"\b(?:his|her|their|he|she|hän|hänen|he|heidän)\b")
RESPONSIBILITIES = re.compile(r"\b(?:responsibilit\w*|duties|work tasks|työtehtäv\w*|vastuu\w*)\b")
ROLE = re.compile(r"\b(?:role|job title|department|toimenkuva|tehtävänimike|yksikkö)\w*\b")
UNAVAILABLE = re.compile(
    r"\b(?:phone\w*|private|personal email|birthday|home address|gmail|salary|office hours|specific courses|"
    r"courses? (?:does|did)|puhelin\w*|henkilökohtai\w*|kotiosoite|syntymä\w*|palkka\w*|vastaanotto\w*|kursseja)\b"
)
STAFF_FOLLOWUP = re.compile(r"\b(?:what does (?:he|she|they) do|where does (?:he|she|they) work|mitä hän tekee|missä hän työskentelee)\b")
NAME_REFERENCE = re.compile(
    r"(?:who is|(?:his|her|their) name is|kuka on|hänen nimensä on|"
    r"does|what does|työskenteleekö|mitä tekee) "
    r"(?P<name>[a-zäöå'-]+(?:[, ]+[a-zäöå'-]+){0,2}?)"
    r"(?: (?:work |do )?at savonia| savoniassa)?"
)
NON_PERSON = re.compile(r"\b(?:the|my|your|a|an|it|savonia|university|yliopisto|ammattikorkeakoulu)\b")
WHICH_PERSON = {"en": "Which person do you mean? Please give their full name.",
                "fi": "Ketä tarkoitat? Kerro henkilön koko nimi."}
NO_PROFILE = {"en": "No personal profile is selected. Please tell me who you mean.",
              "fi": "Profiilia ei ole valittu. Kerro, ketä tarkoitat."}
MISSING_PROFILE = {"en": "The selected profile does not contain that information.",
                   "fi": "Valitussa profiilissa ei ole tätä tietoa."}
MISSING_STAFF = {"en": "That information is unavailable in the verified staff record.",
                 "fi": "Tietoa ei ole saatavilla vahvistetussa henkilöstötietueessa."}


def load_profile(name):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Use a local profile name containing only letters, digits, underscores, or hyphens")
    return json.loads((PROFILE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def personal_field(question):
    return next((field for field, pattern in PERSONAL.items() if re.search(pattern, question)), None)


def speak_work_email(reply, lang):
    def spoken(match):
        separator, at = (" dot ", " at ") if lang == "en" else (" piste ", " ät ")
        local, _ = match[0].split("@")
        return separator.join(local.split(".")) + at + "savonia" + separator + "F I"

    return EMAIL.sub(spoken, reply)


def local_reply(reply, route, *, provenance=None):
    return KnowledgeContext(reply, route, provenance=provenance)


class SavoniaPeople:
    def __init__(self, profile):
        self.profile = profile
        self.path = Path(__file__).with_name("savonia_staff.json")
        self.staff = json.loads(self.path.read_text(encoding="utf-8"))

    def _profile_answer(self, field, query, lang, state):
        if self.profile is None:
            return local_reply(NO_PROFILE[lang], "no profile selected")
        if field == "supervisor":
            if any(word in query for word in ("thesis", "opinnäyte")):
                key = "thesis"
            elif any(word in query for word in ("internship", "harjoittelu")):
                key = "internship"
            else:
                return local_reply("Do you mean your thesis or internship supervisor?" if lang == "en" else
                                   "Tarkoitatko opinnäytetyösi vai harjoittelusi ohjaajaa?", "clarification: supervisor")
            record = self.profile.get("supervisors", {}).get(key, {})
            person = record.get("staff_id")
            if person not in self.staff:
                return local_reply(MISSING_PROFILE[lang], "missing profile fact", provenance="user-supplied profile")
            state.person = person
            name = self.staff[person]["name"]
            label = {"thesis": "opinnäytetyösi", "internship": "harjoittelusi"}[key]
            value = (f"Your profile lists {name} as your {key} supervisor." if lang == "en" else
                     f"Profiilisi mukaan {label} ohjaaja on {name}.")
            if EMAIL_QUERY.search(query):
                return self._staff_answer(person, query, lang)
        elif field == "full_name":
            name = self.profile.get(field)
            value = (f"Your selected profile lists your full name as {name}." if lang == "en" else
                     f"Valitun profiilin mukaan koko nimesi on {name}.") if name else None
        else:
            value = self.profile.get(field, {}).get(lang)
        return local_reply(value or MISSING_PROFILE[lang], f"matched topic: profile {field}", provenance="user-supplied profile")

    def _staff_answer(self, person, query, lang):
        record = self.staff[person]
        refs = tuple((url, record["verified_on"]) for url in record["sources"])
        if UNAVAILABLE.search(query):
            reply = MISSING_STAFF[lang]
        elif EMAIL_QUERY.search(query):
            address = record["work_email"]
            if not address:
                supervisors = (self.profile or {}).get("supervisors", {})
                supplied = next((entry.get("work_email") for entry in supervisors.values()
                                 if entry.get("staff_id") == person and entry.get("work_email")), None)
                if supplied and EMAIL.fullmatch(supplied):
                    reply = (f"The selected profile lists {supplied} as the work email. Savonia's public sources did not independently verify it."
                             if lang == "en" else f"Valitussa profiilissa työsähköpostiksi on merkitty {supplied}. Osoitetta ei ole vahvistettu Savonian julkisista lähteistä.")
                    return local_reply(reply, "matched topic: supplied work email", provenance="user-supplied profile")
                reply = MISSING_STAFF[lang]
            else:
                reply = ("Work email: " if lang == "en" else "Työsähköposti: ") + address + "."
        elif RESPONSIBILITIES.search(query):
            reply = (record["responsibilities"] or {}).get(lang) or MISSING_STAFF[lang]
        else:
            reply = f"{record['name']} — {record['title'][lang]}, {record['department'][lang]}."
            if not ROLE.search(query) and record["responsibilities"]:
                reply += " " + record["responsibilities"][lang]
        return KnowledgeContext(reply, "matched topic: staff" if reply != MISSING_STAFF[lang] else "missing staff fact",
                                refs, fact_ids=(person,))

    def _exact(self, query):
        normalized = fold(query)
        return [person for person, record in self.staff.items() if any(
            re.search(rf"\b{re.escape(part)}(?:n|a|lle|lla|sta)?\b", normalized)
            for part in fold(record["name"]).split()
        )]

    def _suggest(self, name, question, previous, active):
        if len(name) < 4:
            return ()
        candidates = []
        supervisors = (self.profile or {}).get("supervisors", {})
        for person, record in self.staff.items():
            evidence = []
            if person == previous:
                evidence.append("recent person topic")
            for role, entry in supervisors.items():
                if entry.get("staff_id") == person and (person == previous or (active and active.entity == "Savonia")):
                    if person != previous:
                        evidence.append("recent Savonia topic")
                    evidence.append(f"user-supplied {role} supervisor relationship")
            if not evidence:
                continue
            details = candidate_evidence(name, (record["name"], *record["name"].split()), evidence)
            if details:
                candidates.append(Interpretation(person, details, question))
        if self.profile and (full_name := self.profile.get("full_name")):
            details = candidate_evidence(name, (full_name, *full_name.split()), ("selected user-supplied profile",))
            if details:
                candidates.append(Interpretation("profile", details, question))
        return tuple(candidates)

    def _person_name(self, entity):
        return self.profile["full_name"] if entity == "profile" else self.staff[entity]["name"]

    def _display(self, candidates):
        return tuple(replace(candidate, entity=self._person_name(candidate.entity)) for candidate in candidates)

    def _profile_identity(self, lang):
        name = self.profile["full_name"]
        reply = (f"The selected profile identifies this person as {name}." if lang == "en" else
                 f"Valitun profiilin mukaan henkil?n nimi on {name}.")
        return local_reply(reply, "matched topic: profile identity", provenance="user-supplied profile")

    def _clarify(self, candidates, lang, state):
        state.person_pending = candidates
        displayed = self._display(candidates)
        names = [candidate.entity for candidate in displayed]
        if len(names) == 1:
            message = f"Do you mean {names[0]}?" if lang == "en" else f"Tarkoitatko henkilöä {names[0]}?"
        else:
            choices = (" or " if lang == "en" else " vai ").join(names)
            message = f"Which teacher do you mean: {choices}?" if lang == "en" else f"Ketä tarkoitat: {choices}?"
        return KnowledgeContext(message, "clarification: person", candidates=displayed)

    def lookup(self, question, lang, state):
        query = question.strip().casefold()
        bare = query.rstrip(".!?")
        previous, state.person = state.person, None
        pending, state.person_pending = state.person_pending, ()
        people = self._exact(query)
        if pending:
            if REJECT.fullmatch(bare):
                return KnowledgeContext(WHICH_PERSON[lang], "person confirmation rejected",
                                        candidates=self._display(pending))
            chosen = [candidate for candidate in pending if candidate.entity in people]
            if CONFIRM.fullmatch(bare):
                chosen = pending
                if len(chosen) > 1:
                    return self._clarify(pending, lang, state)
            # A bare name selects a candidate. A new question keeps its own intent.
            elif not re.fullmatch(r"[\w'-]+(?: [\w'-]+)?", bare) or len(people) != 1:
                chosen = []
            if len(chosen) == 1:
                candidate = chosen[0]
                if candidate.entity == "profile":
                    answer = self._profile_identity(lang)
                else:
                    state.person = candidate.entity
                    answer = self._staff_answer(state.person, candidate.original_question.casefold(), lang)
                confirmed = replace(candidate, evidence=(*candidate.evidence, "user confirmation"))
                return replace(answer, interpretation=self._display((confirmed,))[0])
        if len(people) > 1 or (not people and TEACHER.search(query)):
            candidates = tuple(Interpretation(person, ("exact name" if people else "ambiguous teacher reference",), question)
                               for person in (people or self.staff))
            return self._clarify(candidates, lang, state)
        if people:
            state.person = people[0]
            return self._staff_answer(state.person, query, lang)
        if field := personal_field(query):
            return self._profile_answer(field, query, lang, state)
        if REFERENT.search(query) and any(pattern.search(query) for pattern in
                                          (EMAIL_QUERY, RESPONSIBILITIES, ROLE, UNAVAILABLE, STAFF_FOLLOWUP)):
            if previous:
                state.person = previous
                return self._staff_answer(previous, query, lang)
            return local_reply("Whose staff details do you mean?" if lang == "en" else
                               "Kenen henkilöstötietoja tarkoitat?", "clarification: person")
        reference = NAME_REFERENCE.fullmatch(bare)
        if reference and not NON_PERSON.search(reference["name"]):
            if bare.startswith(("does ", "what does ", "työskenteleekö ", "mitä tekee ")) and not bare.endswith(
                (" at savonia", " savoniassa")
            ):
                return None
            if self.profile and (full_name := self.profile.get("full_name")) and compact(reference["name"]) in {
                compact(full_name), compact(full_name.split()[-1])
            }:
                return self._profile_identity(lang)
            candidates = self._suggest(reference["name"], question, previous, state.active)
            if candidates:
                return self._clarify(candidates, lang, state)
            return local_reply(WHICH_PERSON[lang], "clarification: unidentified person")
        return None
