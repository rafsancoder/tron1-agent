"""Run a text-only TRON 1 conversation through a local Ollama model."""

import argparse
import re

from scripts.savonia_knowledge import KnowledgeState
from scripts.savonia_people import NO_PROFILE, personal_field


LANGUAGES = {"en": "English", "fi": "Finnish"}


def spoken_reply(reply):
    # Join the stream before filtering so tags split across deltas cannot escape.
    reply = re.sub(r"<think\s*>.*?</think\s*>", "", reply, flags=re.IGNORECASE | re.DOTALL)
    reply = re.sub(r"<\|[a-zA-Z0-9_]+\|>", "", reply)
    # Fail closed on incomplete or unfamiliar markup, including stray closing tags.
    if "<" in reply or ">" in reply:
        raise RuntimeError("The model returned incomplete or unrecognized markup")
    return reply.strip()


class TextConversation:
    def __init__(self, llm, *, recorded_audio=False, knowledge=None, routing_diagnostics=False,
                 personal_questions=False):
        self.llm = llm
        self.knowledge = knowledge
        self.knowledge_state = KnowledgeState() if knowledge is not None else None
        self.routing_diagnostics = routing_diagnostics
        self.personal_questions = personal_questions
        self.history = []
        self.last_reply_source = None
        self.capabilities = (
            "This session transcribes recorded audio and synthesizes spoken replies. "
            "It has no live microphone, robot controls, or external information access."
            if recorded_audio else
            "This session has no microphone, speech output, robot controls, or external information access."
        )

    def reset(self):
        self.history.clear()
        if self.knowledge_state is not None:
            self.knowledge_state = KnowledgeState()
        self.last_reply_source = None

    def ask(self, question, language):
        self.last_reply_source = None
        if language not in LANGUAGES:
            raise ValueError("Expected an ASR language code of en or fi.")
        if not question.strip():
            raise ValueError("Enter a question after the language code.")

        user = {"role": "user", "content": question}
        if self.knowledge is None and self.personal_questions and personal_field(question.casefold()):
            self.last_reply_source = "knowledge"
            if self.routing_diagnostics:
                print("Routing: no profile selected", flush=True)
            return NO_PROFILE[language]
        if self.knowledge is not None:
            context = self.knowledge.lookup(question, language, self.knowledge_state)
            if context is not None:
                if self.routing_diagnostics:
                    print(f"Routing: Savonia {context.route}", flush=True)
                    entity = context.interpretation
                    print(f"  raw ASR transcript: {question!r}", flush=True)
                    print(f"  proposed entity: {entity.entity if entity else 'none'} | "
                          f"evidence: {', '.join(entity.evidence) if entity else 'none'} | "
                          f"fact IDs: {', '.join(context.fact_ids) or 'none'}", flush=True)
                    for candidate in context.candidates:
                        print(f"  candidate: {candidate.entity} | evidence: {', '.join(candidate.evidence)}", flush=True)
                    interpretations = (*context.candidates, entity) if entity else context.candidates
                    if any("similarity" in item for candidate in interpretations for item in candidate.evidence):
                        print("  Similarity scores are heuristics, not calibrated confidence.", flush=True)
                for url, date in context.references:
                    print(f"Savonia source (verified {date}): {url}", flush=True)
                if context.provenance:
                    print(f"Source: {context.provenance}", flush=True)
                self.history.extend([user, {"role": "assistant", "content": context.reply}])
                self.last_reply_source = "knowledge"
                return context.reply
        if self.routing_diagnostics:
            print("Routing: general conversation", flush=True)
            if self.knowledge is not None:
                print(f"  raw ASR transcript: {question!r}", flush=True)
                print("  proposed entity: none | evidence: none | fact IDs: none", flush=True)

        system = {
            "role": "system",
            "content": (
                "You are TRON 1, a conversational assistant. "
                "When asked your name, use the exact name TRON 1. "
                f"Reply in {LANGUAGES[language]} in one or two concise, natural sentences, even if earlier turns use another language. "
                "For Finnish, write idiomatic standard Finnish with correct inflection and possessive endings. "
                "For a request to repeat, rephrase, or translate, restate your most recent answer in the current reply language "
                "unless the user points to a different earlier turn. Do not echo the request. "
                "Use names and other details the user supplied earlier in this conversation when relevant. "
                "If the question is unclear, ask a clarifying question politely instead of guessing the user's intent. "
                "Ask about the missing detail without commenting on the user's wording or grammar. "
                "Never infer the user's location from their language. For nearby places or current conditions, "
                "say what information is missing or unavailable, without inventing places or facts. "
                "Do not claim to have searched, retrieved live data, or performed physical actions. "
                "Describe your capabilities for this session accurately: you can read and reply to text in English and "
                f"Finnish and refer to earlier turns in this conversation. {self.capabilities} "
                "Return only the final reply, with no reasoning or control tokens."
            ),
        }
        if self.knowledge is not None:
            system["content"] += (
                " For Savonia-specific claims, rely only on verified Savonia facts supplied in this conversation. "
                "Say when the local knowledge file does not contain the answer. "
                "Ask which institution is meant when the reference is unclear. Do not guess founders, counts, rankings or locations. "
                "Never claim to have searched a database, checked a website, or performed a live lookup."
            )
        reply = spoken_reply("".join(chunk for chunk in self.llm.response([system, *self.history, user]) if chunk))
        if not reply:
            raise RuntimeError("The model returned an empty reply. Check the local service and model name.")
        self.history.extend([user, {"role": "assistant", "content": reply}])
        self.last_reply_source = "llm"
        return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Name of an installed Ollama model")
    parser.add_argument("--savonia", action="store_true", help="Enable local Savonia facts and public staff records")
    parser.add_argument("--profile", help="Select a Git-ignored local profile by name; requires --savonia")
    parser.add_argument("language", nargs="?", choices=LANGUAGES, help="ASR language code")
    parser.add_argument("question", nargs="?", help="Question, quoted if it contains spaces")
    args = parser.parse_args()
    if args.profile and not args.savonia:
        parser.error("--profile requires --savonia")
    if (args.language is None) != (args.question is None):
        parser.error("Provide both language and question, or omit both for an interactive conversation.")

    try:
        from bailing.llm import OpenAILLM
    except ModuleNotFoundError as error:
        if error.name != "openai":
            raise
        parser.error("The OpenAI adapter requires openai==1.76.0 from requirements.txt.")

    knowledge = None
    if args.savonia:
        from scripts.savonia_knowledge import SavoniaKnowledge
        from scripts.savonia_people import load_profile

        try:
            knowledge = SavoniaKnowledge(profile=load_profile(args.profile) if args.profile else None)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    conversation = TextConversation(OpenAILLM({
        "model_name": args.model,
        "api_key": "ollama",
        "url": "http://127.0.0.1:11434/v1",
    }), knowledge=knowledge, personal_questions=True)
    if args.question is not None:
        try:
            print(conversation.ask(args.question, args.language))
        except (ValueError, RuntimeError) as error:
            parser.exit(1, f"{error}\n")
        return

    print("Enter each turn as en <question> or fi <question>. Type /reset to clear history or /quit to exit.")
    while True:
        try:
            turn = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if turn == "/quit":
            return
        if turn == "/reset":
            conversation.reset()
            print("Conversation history and topic cleared.")
            continue
        parts = turn.split(maxsplit=1)
        if len(parts) != 2:
            print("Enter en <question> or fi <question>.")
            continue
        language, question = parts
        try:
            print(conversation.ask(question, language))
        except (ValueError, RuntimeError) as error:
            print(error)


if __name__ == "__main__":
    main()
