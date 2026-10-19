# Chatbot Service (RAG)

FastAPI microservice implementing the RAG chatbot of the paper cited below. It
runs in two phases: an offline index build at startup, and an online per-message
pipeline. The stage names below follow the module descriptions in the paper.

It is the chatbot service of the [AQVAA platform](https://github.com/fsvbach/AQVAA-Platform):
the platform's backend forwards chat requests to it, and it reads the smartvote
data (questions, candidates, reactions and comments) from the platform, so that
data is uploaded once, into the platform's database.

## Paper

> Báez Santamaría, S., Salvisberg, P., Bachmann, F.: Comparing Dialogue Management
> Strategies with a Modular Platform for Conversational Agent Voting Advice
> Applications. In: CONVERSATIONS 2026. Lecture Notes in Computer Science,
> Springer (2026).

The prompts as deployed in the user study, the module benchmarks and the study
data are in the companion repository
[conversations-cavaa-paper](https://github.com/fsvbach/conversations-cavaa-paper).

## Setup

1. Start the AQVAA platform first. The chatbot joins its `aqvaa` Docker network.
2. Upload the question, candidate and reaction batches to the platform (admin UI
   or the platform's setup scripts). Question indices must be numeric (smartvote IDs).
3. Create the `.env` and fill it in:
    ```bash
    cp .env.example .env
    ```
    - `LLM_SERVICE_TOKEN`: the same value as `LLM_SERVICE_TOKEN` in the platform's `.env`
    - `LLM_REACTION_BATCH_ID`: the ID of the reaction batch to index
    - `CHATBOT_DATA_DIR`: the data directory (see below)
    - `LLM_*`: the model endpoints
4. Build and start:
    ```bash
    docker compose up --build -d
    ```
5. In the platform's `.env`, set `LLM_SERVICE_URL=http://chatbot:8090` and restart its backend.

Candidate data is fetched only when the Milvus collections are empty. After
switching to another reaction batch, restart once with `LLM_FORCE_REINDEX=true`.

## Data directory

The chatbot's knowledge besides the platform data is not part of this repo. It is
mounted read-only from `CHATBOT_DATA_DIR` at `/app/data`:

```
<CHATBOT_DATA_DIR>/
├── party_manifestos/<party>/*.pdf|*.md   # one folder per party (see rag/parties.py)
├── additional_question_info.json         # per-question background, pros/cons
└── study_context/<mode>.md               # assistant.md, interview.md
```

Every part is optional; a missing one is logged and skipped. When running
`main.py` on the host instead of in Docker, point `LLM_MANIFESTOS_PATH`,
`LLM_ADDITIONAL_QUESTION_INFO_PATH` and `LLM_STUDY_CONTEXT_PATH` at these files.

## Offline phase — index construction (runs once at startup)

Entry point: `rag/indexer.py` (`build_index`, called from `main.py` lifespan).
Builds three Milvus collections:

- Candidate positions      — `rag/platform_loader.py` (`load_platform_documents`; one reaction batch
  fetched from the platform's `GET /api/llm/reaction-batches/:id/export`)
- Party candidate averages — `rag/platform_loader.py` (same fetch, averaged per party and question)
- Party manifestos/docs    — `rag/party_documents_loader.py` (`load_manifesto_documents`; accepts both
  PDF and Markdown, hierarchical chunking)

Supplementary per-question info (background, pros/cons from the smartvote
popups) has no counterpart in the platform and is read from
`additional_question_info.json` in the data directory (`rag/question_info.py`).

Party documents live in one subfolder per party under `data/party_manifestos/`,
and both file types are picked up: PDFs are converted to Markdown via Docling,
while `.md` files (hand-saved party website pages) are read directly. From that
point on both go through the same heading-based chunker, so the source format
does not affect how a document is indexed.

Embedding: `rag/embedder*.py`.  Vector store: `rag/vector_store.py`.

`rag/indexer.py` owns both halves of the index lifecycle: offline construction
(`build_index`) and the online query entry points (`query_candidates`,
`query_manifestos`, `query_party_candidate_averages`) that the Retriever calls into.

## Online phase — per user message

Pipeline stages (see the Modules section of the paper):

1. Query Rewriter  — `rag/rewriter.py`           (contextualize message -> self-contained query)
2. Guardrail       — `rag/guardrail.py`          (4 parallel checks; canned reply on violation)
3. Query Router    — `rag/query_router.py`       (LLM picks indices + filters -> RetrievalPlan)
4. Study context   — `rag/query_router.py` (`infer_study_context_needed`) decides whether the
                     study description is injected; text from `miscellaneous/study_context.py`
5. Retriever       — `rag/retriever.py`          (embeds query, runs plan concurrently, balances manifestos across parties)
6. Generator       — `rag/assistant_generator.py`   (Assistant mode)
                   — `rag/interview_generator.py`   (Interview mode; stateful phases via `rag/dialogue_state.py`)
   Both stream tokens via `llm/openai.py` (`LLMRole.GENERATION`).
7. Stance classifier — `rag/stance_classifier.py`  (labels the user's own stance; metadata only)

**Note:** this numbering is a simplification for readability — the stages do not
run strictly one after another. Guardrail checks,
query embedding, query routing and the study-context decision then all run
concurrently in a single `asyncio.gather` inside `rag/retriever.py`, and the
whole retrieval stage runs concurrently with stance classification inside the
generators. Only the guardrail result is a hard gate: on a violation the
pipeline returns a predefined reply and skips retrieval and generation entirely.

Shared: 
- `rag/context.py` (context formatting), 
- `rag/parties.py`, 
- `rag/question_info.py`.

API routes: `api/router.py`.

Config: `config.py` (.env, see `.env.example`).

Prompts: `prompts/`.  

### Study context injection

Studies might differ in survey flow and topics of the underlying questions,
so that description is kept out of the code. `infer_study_context_needed`
(`rag/query_router.py`) asks the LLM whether the rewritten query is about the
study/survey itself; if so, `miscellaneous/study_context.py` loads the text and
the retriever passes it on to the generator. If the retrieval plan is otherwise
empty, the study context alone becomes the context for the answer.

The text lives in `study_context/<mode>.md` in the data directory — one file per
`ChatbotMode`, i.e. `assistant.md` and `interview.md`. **To adapt the chatbot to a
new study, edit those Markdown files; no code change is needed.** Deleting or
emptying a file drops the study-context section from the prompt. The directory
is configurable via `LLM_STUDY_CONTEXT_PATH`.

### Stance classification

`rag/stance_classifier.py` labels the user's own position on the VAA statement
(`agree`, `rather_agree`, `rather_disagree`, `disagree`, `irrelevant`) from
their latest message plus the last few turns. It runs under
`LLMRole.CLASSIFICATION`, which resolves to the main generation model — only
guardrails and query routing sit on the auxiliary endpoint (`_AUX_ROLES` in
`config.py`) — and reads token logprobs, so it returns both a label and a
probability distribution over the labels.

It does **not** feed the answer: both generators launch it concurrently with
retrieval and write the result into the response metadata (`stance`,
`stance_probs`), where it is persisted for analysis. On timeout or a malformed
reply it falls back to `irrelevant` so it can never block a response.
