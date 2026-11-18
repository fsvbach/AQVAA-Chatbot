INITIAL_MESSAGES: dict[str, dict[str, str]] = {
    "32214": {
        "de": (
            "Hallo, ich würde heute gerne mit Ihnen über vier verschiedene politische Themen sprechen. "
            "Es wird immer wieder diskutiert, ob das momentane Pensionsalter von 65 Jahren erhöht werden sollte. "
            "Was interessiert Sie an diesem Thema?"
        ),
        "en": (
            "Hello, today I'd like to talk with you about four different political topics. "
            "There is ongoing debate about whether the current retirement age of 65 should be raised. "
            "What interests you about this topic?"
        ),
    },
    "32227": {
        "de": (
            "Es wird oft darüber debattiert, wie man im Schweizer Bildungssystem sicherstellen kann, dass alle die gleichen Bildungschancen erhalten. "
            "Was interessiert Sie an diesem Thema?"
        ),
        "en": (
            "There is often debate about how the Swiss education system can ensure that everyone has the same educational opportunities. "
            "What interests you about this topic?"
        ),
    },
    "32249": {
        "de": (
            "In der Schweiz wird immer wieder darüber diskutiert, ob das Bauverbot für neue Atomkraftwerke (AKWs) aufgehoben werden soll. "
            "Was interessiert Sie an diesem Thema?"
        ),
        "en": (
            "In Switzerland, there is ongoing debate about whether the ban on building new nuclear power plants should be lifted. "
            "What interests you about this topic?"
        ),
    },
    "32269": {
        "de": (
            "In der Politik wird oft darüber diskutiert, welche Beziehung die Schweiz zur Europäischen Union (EU) haben sollte. "
            "Was interessiert Sie an diesem Thema?"
        ),
        "en": (
            "There is frequent political debate about what kind of relationship Switzerland should have with the European Union (EU). "
            "What interests you about this topic?"
        ),
    },
}


def get_initial_message(question_index: int | str, language: str) -> str | None:
    messages = INITIAL_MESSAGES.get(str(question_index))
    if not messages:
        return None
    return messages.get(language) or messages.get("de")
