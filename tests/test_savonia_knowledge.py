from unittest.mock import Mock

import pytest

from scripts.savonia_knowledge import SavoniaKnowledge
from scripts.verify_live import load_knowledge, run_session
from scripts.verify_text import TextConversation


@pytest.mark.parametrize(("question", "lang"), [
    ("Where are Savonia's campuses?", "en"),
    ("Missä Savonian kampukset sijaitsevat?", "fi"),
])
def test_campus_answer_uses_verified_facts_and_prints_sources(question, lang, capsys):
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    reply = conversation.ask(question, lang)
    assert all(city in reply for city in (("Kuopio", "Iisalmi", "Varkaus") if lang == "en"
                                          else ("Kuopiossa", "Iisalmessa", "Varkaudessa")))
    assert "https://" not in reply
    assert "https://www.savonia.fi/en/contact/campuses/visit-us/" in capsys.readouterr().out
    assert conversation.history[0]["content"] == question
    llm.response.assert_not_called()


@pytest.mark.parametrize(("question", "lang", "expected"), [
    ("What time does Savonia open today?", "en", "does not contain"),
    ("Missä huoneessa Savonian luento pidetään tänään?", "fi", "ei sisällä"),
    ("Can I study at Savona?", "en", "Do you mean Savonia University of Applied Sciences in Kuopio?"),
    ("Voiko Savonassa opiskella?", "fi", "Tarkoitatko Savonia-ammattikorkeakoulua Kuopiossa?"),
])
def test_missing_or_uncertain_information_preserves_question(question, lang, expected):
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    assert expected in conversation.ask(question, lang)
    assert conversation.history[0]["content"] == question
    llm.response.assert_not_called()


def test_city_alone_and_robot_identity_still_use_llm():
    llm = Mock()
    llm.response.side_effect = [["My name is TRON 1."], ["Kuopio is a city."]]
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    assert conversation.ask("What is your name?", "en") == "My name is TRON 1."
    assert conversation.ask("Tell me about Kuopio.", "en") == "Kuopio is a city."
    assert llm.response.call_count == 2


def test_confirmation_routes_previous_unmodified_question():
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    raw = "Where are Savona's campuses?"
    assert conversation.ask(raw, "en").startswith("Do you mean")
    assert "Varkaudessa" in conversation.ask("Kyllä.", "fi")
    assert [turn["content"] for turn in conversation.history if turn["role"] == "user"] == [raw, "Kyllä."]
    llm.response.assert_not_called()


def test_unlisted_slogan_does_not_treat_identity_overview_as_answer():
    context = SavoniaKnowledge().lookup("What is Savonia's slogan?", "en")
    assert "does not contain" in context.reply


def test_after_grounded_answer_unavailable_follow_up_stays_local():
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    assert "Savonia-ammattikorkeakoulu" in conversation.ask("What is Savonia's Finnish name?", "en")
    assert "does not contain" in conversation.ask("What time does it open?", "en")
    llm.response.assert_not_called()


@pytest.mark.parametrize("raw", [
    "Sabunya University",
    "Sabo Nia University of Applied Science ... Coopio ... Finland",
])
def test_observed_phonetic_variants_require_confirmation(raw, capsys):
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge(), routing_diagnostics=True)
    assert conversation.ask(raw, "en") == "Do you mean Savonia University of Applied Sciences in Kuopio?"
    assert "clarification" in capsys.readouterr().out
    assert "Savonia-ammattikorkeakoulu" in conversation.ask("Kyllä, tarkoitan sitä.", "fi")
    assert "matched topic: identity" in capsys.readouterr().out
    assert conversation.history[0]["content"] == raw
    llm.response.assert_not_called()


@pytest.mark.parametrize("raw", [
    "Savonia University of Applied Sciences",
    "Sabo Nia University ... spelling is S-A-V-O-N-I-A",
    "Sabo Nia University ... spelling is S A V O N I A",
])
def test_exact_and_spelled_names_produce_facts(raw, capsys):
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge(), routing_diagnostics=True)
    assert "Finnish higher education institution" in conversation.ask(raw, "en")
    assert "matched topic: identity" in capsys.readouterr().out
    assert conversation.history[0]["content"] == raw
    llm.response.assert_not_called()


@pytest.mark.parametrize("raw", ["Sabo Nia Robotics", "Sabunya", "Tell me about Coopio", "Tell me about Savona, Italy"])
def test_unrelated_names_do_not_trigger_the_provider(raw, capsys):
    llm = Mock()
    llm.response.return_value = ["General reply"]
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge(), routing_diagnostics=True)
    assert conversation.ask(raw, "en") == "General reply"
    assert "general conversation" in capsys.readouterr().out
    assert "Never claim to have searched" in llm.response.call_args.args[0][0]["content"]


def test_startup_flag_and_live_reset_clear_pending_confirmation(capsys):
    assert load_knowledge(False) is None
    provider = load_knowledge(True)
    status = capsys.readouterr().out
    assert "disabled" in status and "enabled" in status
    assert "savonia_facts.json" in status and "5 facts" in status
    llm = Mock()
    llm.response.return_value = ["General reply"]
    conversation = TextConversation(llm, knowledge=provider)
    conversation.ask("Sabunya University", "en")
    commands = iter(["/reset", "/quit"])
    run_session(None, None, conversation, None, 8, read=lambda _: next(commands))
    assert conversation.history == []
    assert conversation.ask("Kyllä.", "fi") == "General reply"


def test_confirmation_answers_original_topic_and_reports_missing_fact(capsys):
    conversation = TextConversation(Mock(), knowledge=SavoniaKnowledge(), routing_diagnostics=True)
    conversation.ask("Where is Sabunya University?", "en")
    assert "Varkaudessa" in conversation.ask("Kyllä.", "fi")
    assert "matched topic: campuses" in capsys.readouterr().out
    conversation.ask("What time does Savonia open today?", "en")
    assert "missing fact" in capsys.readouterr().out


def test_observed_sabuna_location_requires_confirmation_and_preserves_evidence(capsys):
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge(), routing_diagnostics=True)
    raw = "where is Sabuna University of Applied Science"
    assert conversation.ask(raw, "en").startswith("Do you mean")
    assert conversation.knowledge_state.pending.original_question == raw
    assert "sabuna" in str(conversation.knowledge_state.pending.evidence)
    assert "Kuopiossa" in conversation.ask("Kyllä.", "fi")
    assert conversation.history[0]["content"] == raw
    output = capsys.readouterr().out
    assert raw in output and "fact IDs: campuses" in output and "proposed entity: Savonia" in output
    llm.response.assert_not_called()


def test_explicit_spaced_name_and_confirmed_followups_in_both_languages():
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    raw = "Saboenia University of Applied Science. It's SAVO NIA"
    assert "Finnish higher education institution" in conversation.ask(raw, "en")
    assert "Kuopio" in conversation.ask("Yes, can you tell me where is located?", "en")
    assert "AMK- ja YAMK-tutkintoja" in conversation.ask("Mitä siellä voi opiskella?", "fi")
    assert "Kuopiossa" in conversation.ask("Missä se sijaitsee?", "fi")
    assert "bachelor's and master's" in conversation.ask("What can I study there?", "en")
    assert "Kuopio" in conversation.ask("Where is it located?", "en")
    assert conversation.history[0]["content"] == raw
    llm.response.assert_not_called()


@pytest.mark.parametrize("question", ["co-founder of Saboven University", "How many campuses does Savonia have?",
                                    "What is Savonia's ranking?", "Who founded Savonia?"])
def test_out_of_scope_questions_cannot_reach_general_model(question):
    llm = Mock()
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    reply = conversation.ask(question, "en")
    if "Saboven" in question:
        assert reply.startswith("Do you mean")
        reply = conversation.ask("Yes.", "en")
    assert "does not contain" in reply
    llm.response.assert_not_called()


def test_rejection_and_different_institution_do_not_reuse_savonia():
    llm = Mock()
    llm.response.return_value = ["Generic reply"]
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    conversation.ask("where is Sabuna University of Applied Science", "en")
    assert "Which institution" in conversation.ask("No.", "en")
    assert conversation.knowledge_state.pending is None
    assert conversation.ask("Yes.", "en") == "Generic reply"
    assert "Savonia" in conversation.ask("Savonia", "en")
    assert "no verified information" in conversation.ask("Where is the University of Helsinki located?", "en")
    assert "no verified information" in conversation.ask("Where is it located?", "en")
    assert conversation.ask("Tell me a joke.", "en") == "Generic reply"
    assert "Which place or institution" in conversation.ask("Where is it located?", "en")


def test_similarity_requires_institution_signal_and_recent_context():
    provider = SavoniaKnowledge()
    llm = Mock()
    llm.response.return_value = ["Generic reply"]
    conversation = TextConversation(llm, knowledge=provider)
    assert "Which institution" in conversation.ask("Savonix University", "en")
    conversation.ask("Savonia", "en")
    assert conversation.ask("Savonix Robotics", "en") == "Generic reply"
    conversation.ask("Savonia", "en")
    assert conversation.ask("Savonix University", "en").startswith("Do you mean")


@pytest.mark.parametrize(("raw", "fact"), [
    ("I'm to know about Sabonia University of Applied Science in Located in Co-opio.", "campuses"),
    ("You don't know about Sabhanya University.", "identity"),
    ("Tell me about Sa Boh Nia University.", "identity"),
])
def test_live_school_variants_are_compared_before_fallback(raw, fact):
    provider = SavoniaKnowledge()
    conversation = TextConversation(Mock(), knowledge=provider)
    assert conversation.ask(raw, "en").startswith("Do you mean Savonia")
    assert conversation.knowledge_state.pending.original_question == raw
    assert conversation.ask("Kyllä.", "fi") == provider.facts[fact]["fi"]


def test_unmatched_school_with_campus_city_is_unknown():
    provider = SavoniaKnowledge()
    context = provider.lookup("Tell me about Nebula University in Kuopio.", "en")
    assert context.route == "clarification: unknown institution"
    context = provider.lookup("Tell me about University of Eastern Finland in Kuopio.", "en")
    assert context.route == "missing fact: other institution"
