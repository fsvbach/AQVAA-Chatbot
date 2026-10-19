"""System prompts for all LLM calls in the chatbot pipeline.

Each module keeps the prompt version that was deployed in the user study and
exposes it as ``CURRENT`` (the rewriter: ``ASSISTANT_CURRENT`` and
``INTERVIEW_CURRENT``). Earlier iterations were removed from the working tree;
they remain in the git history. The version names (``v4``, ``de_rules_v4``, ...)
are kept so that evaluation runs can still be traced to a prompt.

Numbering as in the companion paper repository
(https://github.com/fsvbach/conversations-cavaa-paper, ``prompts/chatbot_prompts.md``):

    off_topic_guardrail       Prompt 1
    critical_guardrail        Prompt 2
    vote_intention_guardrail  Prompt 3
    llm_opinion_guardrail     Prompt 4
    classifier_prompts        Prompt 5  (query router)
    query_rewriting_prompt    Prompts 6 and 7
    assistant_system_prompts  Prompt 8
    interview_prompts         Prompts 9 to 13
    stance_prompts            Prompt 14
    study_context_prompt      Prompt 15
"""
