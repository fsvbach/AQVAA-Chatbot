INITIAL_MESSAGES: dict[str, dict[str, str]] = {
    "32214":{
        "de":(
            "Hallo, ich bin ein KI-Bot und würde heute gerne mit Ihnen über vier verschiedene politische Themen sprechen. "
            "Zuerst wollen wir über das Pensionsalter sprechen. Es wird immer wieder diskutiert, ob das momentane Pensionsalter von 65 Jahren erhöht werden sollte. "
            "Haben Sie gerade eine Frage dazu? Möchten Sie zum Beispiel wissen, welche Pro- und Contra-Argumente dazu genannt werden, oder wie verschiedene Parteien zu einer Erhöhung des Pensionsalters stehen? "
        ),
        "en":(
            "Hello, I'm an AI chatbot, and today I'd like to talk with you about four different political topics. "
            "First, let's discuss the retirement age. There is ongoing debate about whether the current retirement age of 65 should be raised. "
            "Do you have a question about this topic? For example, would you like to know the main arguments for and against raising the retirement age, or how different political parties view an increase in the retirement age? "
        )
    },
    "32227":{
        "de": (
            "Lassen Sie uns nun über das Schweizer Bildungssystem sprechen. Es wird oft darüber debattiert, wie man sicherstellen kann, dass alle die gleichen Bildungschancen erhalten. "
            "Haben Sie gerade eine Frage dazu? Möchten Sie zum Beispiel wissen, welche Positionen es zu diesem Thema gibt? "
        ),
        "en": (
            "Now let's talk about the Swiss education system. There's often debate about how to ensure that everyone has the same educational opportunities. "
            "Do you have a question about this right away? For example, would you like to know what positions exist on this topic? "
        ),
    },
    
    "32249":{
        "de":(
            "Die Nutzung von Atomkraftwerken (AKWs) ist ein kontroverses Thema. In der Schweiz wird immer wieder darüber diskutiert, ob das Bauverbot für neue AKWs aufgehoben werden soll. "
            "Möchten Sie mehr über die aktuelle Situation in der Schweiz oder über die wichtigsten Argumente für und gegen den Neubau erfahren?"
        ),
        "en":(
            "The use of nuclear power plants (NPPs) is a controversial topic. In Switzerland, there is ongoing debate about whether the ban on building new nuclear power plants should be lifted. "
            "Would you like to learn more about the current situation in Switzerland or about the key arguments for and against building new nuclear power plants?"
        )
    },


    "32269":{
        "de":( 
            "Zuletzt möchte ich mit Ihnen über die Europäische Union sprechen. Über die Frage, welche Beziehung die Schweiz zu "
            "der EU haben sollte, wird in der Politik oft diskutiert. Dabei geht es oft um die Bilateralen Verträge. "
            'Möchten Sie gerne mehr über *die Bilateralen* erfahren, oder haben Sie sonst gerade eine Frage zur EU?'
            ),
        "en": (
            "Lastly, I would like to talk to you about the European Union. The question of what kind of relationship Switzerland "
            "should have with the EU is frequently debated in politics. This debate often revolves around the Bilateral Agreements. "
            'Would you like to learn more about *the Bilaterals*, or do you have another question about the EU?'
        ),
    },

    "32275": {
        "de": (
            "Lassen Sie uns über die **freie Marktwirtschaft** sprechen. In der Politik wird oft darüber gestritten, "
            "wie viel Freiheit der Markt braucht und wie viel Kontrolle durch den Staat notwendig ist. "
            "Haben Sie dazu gleich eine Frage? Oder soll ich Ihnen erstmal erklären, was man unter der **freien Marktwirtschaft** versteht?"
        ),
        "en": (
            "Let's talk about **free market economy**. In politics, there's often debate about "
            "how much freedom the market needs and how much regulation by the state is necessary. "
            "Do you have a question right away? Or would you like me to first explain what **free market economics** means?"
        ),
    },

    "32280":{
        "de": (
            "Nun sprechen wir über die Rolle der **Wirtschaft** beim **Umweltschutz**. "
            "In der Politik wird oft gestritten, wie viel Verantwortung Unternehmen tragen sollen und welche Rolle der Staat dabei spielt. "
            "Haben Sie gleich eine Frage dazu? Oder soll ich Ihnen kurz erklären, worum es bei diesem Thema geht?"
        ),
        "en": (
            "Now let's talk about the role of **business and the economy** in **environmental protection**. "
            "In politics, there's often debate about how much responsibility companies should bear and what role the state should play. "
            "Do you have a question right away? Or would you like me to briefly explain what this topic is about?"
        ),
    },

    "32281":{
        "de": (
            "Wir sind beim letzten der fünf Themen angelangt: **Soziale Wohlfahrt**. "
            "Dabei geht es um Fragen wie: Wie viel soll der Staat für seine Bürgerinnen und Bürger absichern, und wo liegt die Eigenverantwortung? "
            "Haben Sie gleich eine Frage dazu? Oder soll ich Ihnen erstmal erklären, was man unter **sozialer Wohlfahrt** versteht?"
        ),
        "en": (
            "We've reached the last of the five topics: **social welfare**. "
            "This covers questions like: how much should the state provide for its citizens, and where does personal responsibility begin? "
            "Do you have a question right away? Or would you like me to first explain what **social welfare** means?"
        ),
    },
    
}


def get_initial_message(question_index: int | str, language: str) -> str | None:
    messages = INITIAL_MESSAGES.get(str(question_index))
    if not messages:
        return None
    return messages.get(language) or messages.get("de")
