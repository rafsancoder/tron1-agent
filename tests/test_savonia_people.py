from unittest.mock import Mock

import pytest

from scripts.savonia_knowledge import SavoniaKnowledge
from scripts.savonia_people import speak_work_email
from scripts.verify_text import TextConversation


def test_split_profile_name_is_contextual_and_preserves_intent():
    raw = "Who is rough, son?"
    llm = Mock()
    llm.response.return_value = ["General reply"]
    anonymous = TextConversation(llm, knowledge=SavoniaKnowledge())
    assert anonymous.ask(raw, "en").startswith("Which person")
    provider = SavoniaKnowledge(profile={"full_name": "Example Rofsan"})
    conversation = TextConversation(llm, knowledge=provider)
    assert conversation.ask(raw, "en") == "Do you mean Example Rofsan?"
    assert conversation.knowledge_state.person_pending[0].original_question == raw
    assert "Example Rofsan" in conversation.ask("Kyllä.", "fi")
    assert any(turn["content"] == raw for turn in conversation.history)
    conversation.ask(raw, "en")
    assert conversation.ask("No.", "en").startswith("Which person")
    assert conversation.knowledge_state.person_pending == ()
    for unrelated in ("play a demo", "the sea is rough"):
        assert conversation.ask(unrelated, "en") == "General reply"

@pytest.fixture
def profile():
    return {"full_name": "Example Person", "internship": {"en": "Robot Savo", "fi": "Robot Savo"},
            "supervisors": {"thesis": {"staff_id": "teemu_matilainen", "work_email": "profile.contact@savonia.fi"},
                            "internship": {"staff_id": "markku_kellomaki"}}}


def test_no_profile_never_assumes_a_name_or_supervisor():
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    assert "No personal profile" in conversation.ask("What is my full name?", "en")
    assert "Profiilia ei ole valittu" in conversation.ask("Kuka on opinnäytetyöni ohjaaja?", "fi")
    llm.response.assert_not_called()


def test_profile_fields_and_supervisor_provenance(profile, capsys):
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge(profile=profile))
    assert "Example Person" in conversation.ask("What is my full name?", "en")
    assert "Robot Savo" in conversation.ask("What work am I doing?", "en")
    assert "Teemu Matilainen" in conversation.ask("Kuka on opinnäytetyöni ohjaaja?", "fi")
    answer = conversation.ask("What is his work email?", "en")
    assert "profile.contact@savonia.fi" in answer and "profile" in answer.lower()
    assert "user-supplied profile" in capsys.readouterr().out
    llm.response.assert_not_called()


def test_verified_staff_followup_and_ambiguous_teacher_clear_context(capsys):
    conversation = TextConversation(Mock(), knowledge=SavoniaKnowledge())
    assert "Senior lecturer" in conversation.ask("What does Markku Kellomaki do at Savonia?", "en")
    assert "markku.kellomaki@savonia.fi" in conversation.ask("Mikä hänen sähköpostiosoitteensa on?", "fi")
    assert "https://www.savonia.fi/" in capsys.readouterr().out
    assert "Which teacher" in conversation.ask("What is my teacher's email?", "en")
    assert "Whose" in conversation.ask("What is his work email?", "en")


def test_reset_keeps_profile_but_discards_person_and_missing_email_does_not_guess(profile):
    conversation = TextConversation(Mock(), knowledge=SavoniaKnowledge(profile=profile))
    conversation.ask("Who is my internship supervisor?", "en")
    conversation.reset()
    assert "Whose" in conversation.ask("What is his work email?", "en")
    assert "Example Person" in conversation.ask("What is my full name?", "en")
    public_only = TextConversation(Mock(), knowledge=SavoniaKnowledge())
    assert "unavailable" in public_only.ask("What is Teemu Matilainen's email?", "en")


def test_email_speech_leaves_displayed_address_intact():
    displayed = "Email: markku.kellomaki@savonia.fi."
    assert speak_work_email(displayed, "en") == "Email: markku dot kellomaki at savonia dot F I."
    assert "piste" in speak_work_email(displayed, "fi")
    assert displayed.endswith("markku.kellomaki@savonia.fi.")


def test_unselected_live_profile_does_not_fall_back_to_llm():
    llm = Mock()
    conversation = TextConversation(llm, personal_questions=True)
    assert "No personal profile" in conversation.ask("What work am I doing?", "en")
    llm.response.assert_not_called()


@pytest.mark.parametrize(("role", "raw", "person"), [
    ("internship", "His name is Markle.", "Markku Kellomäki"),
    ("thesis", "Who is demo?", "Teemu Matilainen"),
])
def test_asr_person_candidate_requires_confirmation(profile, role, raw, person, capsys):
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge(profile=profile), routing_diagnostics=True)
    conversation.ask(f"Who is my {role} supervisor?", "en")
    assert conversation.ask(raw, "en") == f"Do you mean {person}?"
    assert conversation.knowledge_state.person_pending[0].original_question == raw
    assert conversation.history[-2]["content"] == raw
    assert person in conversation.ask("Kyllä, tarkoitan sitä.", "fi")
    assert person in conversation.ask("What does he do?", "en")
    assert "@savonia.fi" in conversation.ask("Mikä hänen sähköpostiosoitteensa on?", "fi")
    output = capsys.readouterr().out
    assert "spelling similarity" in output and "sound-key similarity" in output and "user-supplied" in output
    llm.response.assert_not_called()


def test_neutral_clarification_rejection_and_unrelated_demo(profile):
    llm = Mock()
    llm.response.return_value = ["Generic reply"]
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge(profile=profile))
    assert conversation.ask("Who is demo?", "en").startswith("Which person")
    assert conversation.knowledge_state.person_pending == ()
    conversation.ask("Who is my thesis supervisor?", "en")
    assert "Teemu Matilainen" in conversation.ask("Who is demo?", "en")
    assert conversation.ask("Ei.", "fi").startswith("Ketä")
    assert conversation.ask("What is his work email?", "en").startswith("Whose")
    assert conversation.ask("Who is Unknown Person?", "en").startswith("Which person")
    assert conversation.ask("Does Unknown Person work at Savonia?", "en").startswith("Which person")
    assert conversation.ask("play a demo", "en") == "Generic reply"
    llm.response.assert_called_once()


def test_multiple_candidates_require_named_choice_and_answer_original_email(profile):
    conversation = TextConversation(Mock(), knowledge=SavoniaKnowledge(profile=profile))
    reply = conversation.ask("What is Teemu or Markku's work email?", "en")
    assert "Teemu Matilainen" in reply and "Markku Kellomäki" in reply
    assert "Which teacher" in conversation.ask("Yes.", "en")
    assert "markku.kellomaki@savonia.fi" in conversation.ask("Markku", "fi")
    conversation.ask("Who is demo?", "en")
    conversation.reset()
    assert conversation.knowledge_state.person_pending == ()
