"""Deterministic, extractive customer support reference; Python standard library only."""

import json
import re
import sys
from pathlib import Path


class ValidationError(ValueError):
    pass


class Validator:
    """One validation boundary for requests and optional selector results."""

    @staticmethod
    def object(value, required, optional=()):
        if not isinstance(value, dict):
            raise ValidationError("Expected a JSON object")
        if set(value) - set(required) - set(optional) or set(required) - set(value):
            raise ValidationError("Missing or unknown fields")

    @staticmethod
    def text(value, name, maximum=4000):
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise ValidationError(f"{name} must be nonempty text of at most {maximum} characters")
        return value.strip()

    @classmethod
    def request(cls, value):
        cls.object(value, ("question", "knowledge_base", "team_online"))
        question = cls.text(value["question"], "question", 2000)
        if type(value["team_online"]) is not bool:
            raise ValidationError("team_online must be a boolean")
        kb = value["knowledge_base"]
        if not isinstance(kb, list) or len(kb) > 100:
            raise ValidationError("knowledge_base must be a list with at most 100 articles")
        articles, seen = [], set()
        for article in kb:
            cls.object(article, ("id", "title", "body"), ("keywords",))
            item = {key: cls.text(article[key], key, 4000 if key == "body" else 200)
                    for key in ("id", "title", "body")}
            if item["id"] in seen:
                raise ValidationError("Article IDs must be unique")
            seen.add(item["id"])
            keywords = article.get("keywords", [])
            if not isinstance(keywords, list) or len(keywords) > 30:
                raise ValidationError("keywords must be a list with at most 30 entries")
            item["keywords"] = [cls.text(word, "keyword", 100) for word in keywords]
            articles.append(item)
        return {"question": question, "knowledge_base": articles,
                "team_online": value["team_online"]}

    @classmethod
    def selection(cls, value, allowed):
        cls.object(value, ("article_ids",))
        ids = value["article_ids"]
        if (not isinstance(ids, list) or len(ids) > 3
                or any(not isinstance(i, str) for i in ids)):
            raise ValidationError("Selector article_ids must contain at most three strings")
        if len(set(ids)) != len(ids) or any(i not in allowed for i in ids):
            raise ValidationError("Selector must select unique retrieved article IDs")
        return ids


STOPWORDS = set(
    "a an the i me my we our you your is are was were be can could would should "
    "do does did how what when where why to for of in on at and or it this that "
    "please help with have has want need about tell get".split()
)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - STOPWORDS


def support(payload, selector=None):
    """Selector is optional: callable(context) -> {'article_ids': [retrieved IDs]}.

    Selection is injectable, not answer generation: final text is always copied
    from the validated knowledge base. No provider calls or persistent tickets.
    """
    request = Validator.request(payload)
    question = request["question"]
    query = tokens(question)
    ranked = []
    for article in request["knowledge_base"]:
        article_tokens = tokens(" ".join(
            [article["title"], article["body"]] + article["keywords"]))
        overlap = len(query & article_tokens)
        coverage = overlap / len(query) if query else 0
        if overlap and coverage >= 0.5:
            ranked.append((coverage, overlap, article["id"], article))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    candidates = [row[3] for row in ranked[:3]]
    human = bool(re.search(
        r"\b(human|agent|representative|person)\b", question, re.IGNORECASE))
    reason = "human_requested" if human else ("no_grounded_match" if not candidates else None)
    if reason is None and selector is not None:
        context = {"question": question,
                   "candidates": [{"id": a["id"], "title": a["title"], "body": a["body"]}
                                  for a in candidates]}
        try:
            selected = selector(context)
        except Exception as exc:
            raise ValidationError("Optional selector failed") from exc
        ids = Validator.selection(selected, {a["id"] for a in candidates})
        by_id = {a["id"]: a for a in candidates}
        candidates = [by_id[i] for i in ids]
        if not candidates:
            reason = "selector_abstained"
    if reason:
        online = request["team_online"]
        answer = ("A human support contact is the next step."
                  if human else "I could not find a grounded answer in the supplied knowledge base.")
        answer += (" Your support team is marked online; contact them through your usual channel."
                   if online else
                   " Your support team is offline; contact them when they return.")
        answer += " No ticket or message has been sent."
        return {
            "status": "ok", "resolution": "escalated", "answer": answer, "citations": [],
            "handoff": {"needed": True, "reason": reason, "team_online": online,
                        "delivery": "not_sent", "summary": question},
        }
    return {
        "status": "ok", "resolution": "answered",
        "answer": "From the supplied support knowledge base:\n\n" + "\n\n".join(
            f"[{a['id']}] {a['body']}" for a in candidates),
        "citations": [{"id": a["id"], "title": a["title"], "excerpt": a["body"]}
                      for a in candidates],
        "handoff": {"needed": False, "reason": None,
                    "team_online": request["team_online"], "delivery": "not_sent",
                    "summary": None},
    }


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate JSON fields are not allowed")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Non-finite JSON values are not allowed")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValidationError("Usage: python -B implementation.py INPUT.json")
        path = Path(args[0])
        if path.stat().st_size > 1_000_000:
            raise ValidationError("Input file exceeds 1000000 bytes")
        payload = json.loads(path.read_text(encoding="utf-8"),
                             object_pairs_hook=strict_object, parse_constant=reject_constant)
        result = support(payload)
    except (ValidationError, OSError, UnicodeError, ValueError, RecursionError):
        print(json.dumps({"status": "error",
                          "error": "Invalid input, inaccessible file, or invalid selector result."}))
        return 2
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
