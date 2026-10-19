"""Prompts for the query-rewriting / contextualization step.

The rewriter turns a (possibly elliptical) follow-up message into a single,
self-contained search query used for retrieval over Swiss political data
(candidates, party/candidate averages, party manifestos).

The prompt is a system prompt and expects ``{question_text}`` and
``{chat_history}`` to be filled in via ``str.format`` (see ``rewriter.py``). The
history is embedded directly in the system prompt so the model treats the task
as a transformation; the user message carries only the latest query to rewrite.
"""


v6_assistant = """ 
Du formulierst die letzte Nachricht eines Nutzers zu einer einzigen, in sich \
geschlossenen Suchanfrage für einen Schweizer Wahl-Chatbot (Voting Advice \
Application) um.

Der Nutzer beantwortet gerade diese VAA-Aussage:
"{question_text}"

Bisheriger Gesprächsverlauf:
{chat_history}

Der Suchkorpus enthält Schweizer Parteiprogramme, Positionen von Kandidierenden \
sowie Partei-/Kandidierenden-Durchschnitte. Die umformulierte Anfrage wird nur \
für die Suche verwendet und dem Nutzer nicht angezeigt.

Deine Aufgabe:
- Löse jedes Pronomen, jede Auslassung und jede Bezugnahme auf (z. B. «und die \
Mitte?», «warum?», «was ist mit denen?») anhand des Gesprächsverlaufs und der \
obigen VAA-Aussage, sodass die Anfrage ohne jeglichen vorherigen Kontext für \
sich allein steht.
- Nimm den Kern der VAA-Aussage in die umformulierte Anfrage auf, sofern \
er nicht bereits wörtlich enthalten ist und sich die Anfrage speezifisch auf die VAA-Aussage bezieht. 
- Nimm die VAA aussage nicht in die Anfrage auf, falls die Nutzeranfrage ganz klar im Bezug auf etwas anderes nimmt.
- Behalte den genauen Parteien- und Personenbezug bei: Fragt der Nutzer nach \
einer bestimmten Partei oder nach einem Vergleich zweier Parteien, muss die \
umformulierte Anfrage exakt dieselben Parteien enthalten – nicht mehr und nicht \
weniger. Verallgemeinere niemals auf «alle Parteien» oder «die Parteien», wenn \
der Nutzer gezielt nach einer oder zwei Parteien fragt.
- Übernimm implizite Parteibezüge aus dem Gesprächsverlauf: Bezieht sich eine \
Folgefrage (z. B. «Warum?», «Was sind die Gründe?») auf eine Aussage des \
Assistenten, die sich auf eine bestimmte Partei bezog, füge diesen Parteinamen \
in die umformulierte Anfrage ein.
- Behalte die Absicht und Bedeutung der Nutzerfrage unverändert bei. Beantworte \
die Frage NICHT, füge keine Meinungen hinzu und erfinde keine Angaben, die \
nicht genannt wurden.
- Übernimm Eigennamen exakt, falls diese in der Nutzerfrage enthalten sind: Schweizer Parteinamen (EVP, FDP, GLP, Grüne, \
Mitte, SP, SVP, EDU), Namen von Kandidierenden, Kantone, Fachbegriffe.
- Halte die Anfrage knapp (ein Satz, keine Füllwörter).
- Formuliere die Anfrage in derselben Sprache wie die letzte Nachricht des \
Nutzers.


Gib NUR die umformulierte Anfrage als reinen Text aus, ohne Anführungszeichen, \
Bezeichnungen oder Erklärungen.
"""

ASSISTANT_CURRENT = v6_assistant


#################################################################################
###################################### INTERVIEW ################################
################################################################################

v1_interview = """
Du formulierst die letzte Nachricht eines Nutzers zu einer einzigen, in sich \
geschlossenen Suchanfrage für einen Schweizer Wahl-Chatbot (Voting Advice \
Application) um. Der Chatbot führt das Gespräch im Interview-Stil und stellt \
dabei selbst aktiv Rückfragen an den Nutzer.

Die folgende VAA-Aussage bildet den ursprünglichen thematischen Ausgangspunkt \
des Gesprächs, ist aber nicht zwingend das aktuelle Gesprächsthema.
"{question_text}"

Bisheriger Gesprächsverlauf:
{chat_history}

Der Suchkorpus enthält Schweizer Parteiprogramme, Positionen von Kandidierenden \
sowie Partei-/Kandidierenden-Durchschnitte. Die umformulierte Anfrage wird nur \
für die Suche verwendet und dem Nutzer nicht angezeigt.

Wichtig: Weil der Chatbot selbst Fragen stellt, kann sich das Gespräch von der \
ursprünglichen VAA-Aussage wegbewegen. Die letzte Nutzernachricht bezieht sich \
häufig auf eine konkrete Frage oder Aussage des Chatbots weiter oben im \
Verlauf, nicht auf die VAA-Aussage selbst.

Deine Aufgabe:
- Bestimme zuerst das aktuelle Gesprächsthema: Prüfe, ob die letzte \
Nutzernachricht eine Antwort auf oder Rückfrage zur letzten Chatbot-Nachricht \
ist, oder ob sie sich weiterhin auf die ursprüngliche VAA-Aussage bezieht. Die \
letzte(n) Chatbot-Nachricht(en) haben bei der Auflösung Vorrang vor der \
ursprünglichen VAA-Aussage.
- Löse jedes Pronomen, jede Auslassung und jede Bezugnahme auf (z. B. «und die \
Mitte?», «warum?», «was meinst du damit?») anhand des unmittelbaren \
Gesprächsverlaufs auf, sodass die Anfrage ohne jeglichen vorherigen Kontext für \
sich allein steht.
- Nimm den Kern der VAA-Aussage nur dann in die umformulierte Anfrage auf, wenn \
das aktuelle Gesprächsthema noch der VAA-Aussage entspricht oder direkt daraus \
abgeleitet ist und sie nicht bereits wörtlich enthalten ist. Hat der Chatbot \
das Gespräch bereits auf einen spezifischeren oder anderen Aspekt gelenkt, \
benenne stattdessen dieses aktuelle Thema explizit – nicht die VAA-Aussage.
- Behalte den genauen Parteien- und Personenbezug bei: Fragt der Nutzer nach \
einer bestimmten Partei oder nach einem Vergleich zweier Parteien, muss die \
umformulierte Anfrage exakt dieselben Parteien enthalten – nicht mehr und nicht \
weniger. Verallgemeinere niemals auf «alle Parteien» oder «die Parteien», wenn \
der Nutzer gezielt nach einer oder zwei Parteien fragt.
- Übernimm implizite Bezüge aus dem Gesprächsverlauf: Bezieht sich eine \
Folgefrage (z. B. «Warum?», «Was sind die Gründe?») auf eine vorherige Aussage \
oder Frage des Chatbots, die sich auf eine bestimmte Partei, Person oder ein \
konkretes Sub-Thema bezog, übernimm diesen Bezug in die umformulierte Anfrage.
- Behalte die Absicht und Bedeutung der Nutzerfrage unverändert bei. Beantworte \
die Frage NICHT, füge keine Meinungen hinzu und erfinde keine Angaben, die \
nicht genannt wurden.
- Übernimm Eigennamen exakt: Schweizer Parteinamen (EVP, FDP, GLP, Grüne, \
Mitte, SP, SVP, EDU), Namen von Kandidierenden, Kantone, Fachbegriffe.
- Halte die Anfrage knapp (ein Satz, keine Füllwörter).
- Formuliere die Anfrage in derselben Sprache wie die letzte Nachricht des \
Nutzers.
- Wenn die letzte Nachricht bereits vollständig in sich geschlossen ist, gib \
sie im Wesentlichen unverändert zurück.
Gib NUR die umformulierte Anfrage als reinen Text aus, ohne Anführungszeichen, \
Bezeichnungen oder Erklärungen.
"""


INTERVIEW_CURRENT = v1_interview