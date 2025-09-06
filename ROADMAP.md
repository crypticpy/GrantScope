# GrantScope Modernization Roadmap

This document tracks upgrade ideas and organizes them into actionable batches with clear strategy, scope, and acceptance criteria. The goal is to make GrantScope more robust, modern, and interactive while keeping changes safe and iterative.

Last updated: 2025-09-05

## 0) TL;DR — Execution Order

1. Data correctness + stability (preprocessing, schema validation)
2. AI chat UX upgrade (chat UI + streaming) on top 1–2 pages
3. Multipage navigation + layout polish
4. Visual/interaction upgrades (dialogs, popovers, consistent formatting)
5. AI “tools” and context (Candid fetch as tool, better prompts, guardrails)
6. Performance and caching (groupbys, sampling)
7. Secrets, config, and packaging (pinned deps, secrets, .env)
8. Tests, type hints, and CI
9. Docs, examples, and release notes

---

## 1) Current State Summary (audit)

- Streamlit single-file entry (`app.py`), chart functions in `plots/`, data loader in `loaders/`.
- Data: JSON grants, optional upload; preprocessing explodes semicolon-delimited categories and clusters amounts.
- Visuals: Summary, Distribution, Scatter, Heatmap, Word Clouds, Treemaps, and Top Categories.
- AI: LlamaIndex PandasQueryEngine + OpenAI model; per-view prompts; input via sidebar.
- Utilities: CSV/Excel download helpers; Candid CLI fetcher script.

Key risks/opportunities:

- Preprocessing has correctness issues (explode/map usage; dtype-based fillna; repeated splitting in later pages).
- AI UX is not chat-native; no streaming; prompts lack guardrails.
- UX is single-page; filters/downloads are scattered; table formatting inconsistent.
- Secrets hardcoded in fetcher; schema validation is minimal; no tests; deps unpinned.

---

## 2) Batches and Strategy

### Batch 1 — Data correctness and stability

1.1 Fix preprocessing explode logic for paired code/tran columns (single multi-explode pass; preserve row alignment).

1.2 Replace dtype-key `fillna` with column-wise defaults; ensure numeric/categorical consistency.

1.3 Dedup logic: verify dedupe keys; ensure `grouped_df` remains one row per grant.

1.4 Add upload schema validation with helpful error display and a sample template.

1.5 Normalize cross-code usage so downstream pages don’t re-split strings that are already exploded.

Strategy:

- Implement utility functions for splitting/exploding paired columns.
- Add lightweight unit tests with a tiny fixture (`tests/fixtures/sample_min.json`).
- Cache outputs with `st.cache_data` and stable hash keys.

Acceptance criteria:

- All pages render without errors on sample data and an uploaded file.
- Unit tests pass for preprocessing (list columns, explode counts, dedupe invariants).

---

### Batch 2 — AI chat UX upgrade (modern Streamlit)

2.1 Replace selectbox+submit with chat primitives (`st.chat_message`, `st.chat_input`) on Data Summary and Distribution pages.

2.2 Add response streaming with a visible spinner/status and cancel affordance.

2.3 Maintain per-page chat history in `st.session_state` (keyed by page).

2.4 Inject current filters and a compact data sample into the context for grounded answers.

Strategy:

- Wrap LlamaIndex calls with a small adapter that supports streaming and tool-constrained answers.
- Add a shared ChatPanel component and drop-in to pages.

Acceptance criteria:

- Users can converse with the page; messages appear in bubbles; partial tokens stream smoothly.
- Answers explicitly reference current filters and avoid hallucinating unknown fields.

---

### Batch 3 — Multipage navigation and layout

3.1 Convert to multipage app (`pages/` directory): Home (Summary), Distribution, Scatter, Heatmap, Word Clouds, Treemaps, Top Categories, Relationships.

3.2 Persist selected “User Role” globally in session and reflect in page content.

3.3 Group controls with `st.tabs`, `st.popover` for advanced filters, and consistent headers.

Strategy:

- Create one page per existing plot module; thin the entry `app.py` into global state and nav.
- Introduce a minimal layout theme and consistent containers.

Acceptance criteria:

- Each chart has its own page; navigation is clear; state persists across pages.

---

### Batch 4 — Visual polish and interaction upgrades

4.1 Use `st.dialog` for clickthrough grant details (full description, links, metadata).

4.2 Move downloads to a `st.popover` with CSV/Excel options; unify formatting.

4.3 Currency and number formatting across tooltips/axes; consistent color scales.

4.4 Use `st.data_editor` with column configs (currency, URLs) and row selection where sensible.

4.5 Add dark/light theme support via `.streamlit/config.toml` and Plotly template.

Strategy:

- Add a `ui/formatting.py` with helpers (currency formatters, color palettes).
- Refactor chart builders to reuse common formatting.

Acceptance criteria:

- Consistent formatting across pages; dialogs/poppers work; downloads are centralized.

---

### Batch 5 — AI tools and safer prompting

5.1 Introduce structured system prompts with schema, privacy, and “don’t fabricate beyond available columns.”

5.2 Add a “Fetch grants” tool the AI can call (parameters: years, subjects, locations, transaction types) that writes results into session.

5.3 Page-specific context packers (filters, selected entities) to keep answers focused.

5.4 Optional: add summarization snippets for large tables to reduce token usage.

Strategy:

- Define a small tool registry the chat adapter can expose to the LLM.
- Guardrail prompts with consistent headers and page context sections.

Acceptance criteria:

- Tool-invoking questions (e.g., “pull 2018–2020 Texas health grants”) fetch and update data safely with rate-limit awareness.
- Answers never reference unknown columns; violations covered by tests.

---

### Batch 6 — Performance and caching

6.1 Cache heavy groupbys/pivots with `st.cache_data` and explicit keys for filters.

6.2 Add optional server-side sampling for very large scatter plots.

6.3 Avoid redundant recomputations; compute once per page on input changes.

6.4 Use `st.fragment` for slow subcomponents to reduce reruns.

Strategy:

- Introduce `data/aggregates.py` with memoized helpers.
- Add a “Performance” toggle to enable sampling when row counts exceed thresholds.

Acceptance criteria:

- Interactions feel snappy on datasets up to defined limits; no unnecessary reruns.

---

### Batch 7 — Secrets, config, and data access

7.1 Support `st.secrets` for API keys; fallback to sidebar input only if unset.

7.2 Move Candid API key from code to secrets and document it.

7.3 Add `.env` optional support via `python-dotenv`; never commit secrets.

7.4 Add input rate limiting/backoff for the fetcher; parameter validation with clear errors.

Strategy:

- Centralize config in `utils/config.py`; read env, secrets, and defaults.
- Wrap fetcher in functions usable by UI and AI tool.

Acceptance criteria:

- No hardcoded secrets; users can run locally with `secrets.toml` or env vars.

---

### Batch 8 — Packaging and dependencies

8.1 Pin runtime dependencies with compatible ranges (Streamlit, Plotly, pandas, llama-index, wordcloud, numpy).

8.2 Separate `requirements-dev.txt` (pytest, ruff/flake8, mypy, types for requests).

8.3 Add a minimal Makefile/Tasks (Windows-friendly scripts in README) for common flows.

Strategy:

- Bump versions carefully and test; document upgrade steps.

Acceptance criteria:

- Fresh install on a clean environment works; lockfile (optional) consistent.

---

### Batch 9 — Tests, typing, and CI

9.1 Add unit tests for preprocessing, prompt building, and at least one chart helper.

9.2 Add type hints in loaders/utils; run mypy in CI.

9.3 Add lint (ruff/flake8) and format (black) checks.

9.4 GitHub Actions workflow for test/lint on PRs.

Strategy:

- Start with high-signal tests on data transforms; expand with fixtures.

Acceptance criteria:

- CI green on main and PRs; contributors have fast feedback.

---

### Batch 10 — Docs and examples

10.1 Update README with multipage navigation, secrets, and quick-start commands.

10.2 Add a short “User Guide” page in the app with tips and example questions.

10.3 Add a “Developer Guide” (folder structure, how to add a page, tests).

10.4 Provide a larger sample dataset (sanitized) or scripted generator.

Strategy:

- Keep docs close to code; link from Home page.

Acceptance criteria:

- New users can run the app; devs can add a page by following the guide.

---

## 3) Streamlit modern features to leverage

- Chat UI: `st.chat_message`, `st.chat_input` for conversational analysis.
- Dialogs: `st.dialog` for detailed grant views on click.
- Popovers: `st.popover` to house advanced filters and downloads.
- Fragments: `st.fragment` to isolate slow subcomponents and reduce reruns.
- Multipage apps: `pages/` directory with per-page code and shared state.
- Data editor configs: richer `st.data_editor` with column configs (currency, links).
- Caching: `st.cache_data`/`st.cache_resource` with explicit keys.

Note: We’ll target current stable Streamlit supporting these primitives; final min-version will be pinned in Batch 8.

---

## 4) Cross-cutting concerns

- Accessibility: colorblind-friendly palettes, sufficient contrast, aria labels for interactive elements.
- Internationalization-lite: currency formatting and date handling decoupled from logic.
- Privacy: never echo secrets or raw PII to the model; surface a privacy note in AI chat.
- Observability: optional debug panel (log last query, cache hits) behind a toggle.

---

## 5) Rollout plan and risk management

- Feature flags: wrap new chat UI and multipage nav behind toggles initially.
- Progressive enhancement: keep previous UI paths until parity is reached.
- Backups: branch-per-batch with PRs and screenshots/gifs.
- Recovery: maintain simple “compat mode” single-page app until Batch 3 completes.

---

## 6) Work breakdown (granular)

- B1.T1 Refactor `preprocess_data` with pairwise multi-explode helper.
- B1.T2 Add schema validator and sample template; error panel on upload.
- B2.T1 Chat adapter with streaming; shared ChatPanel component.
- B2.T2 Wire chat into Summary + Distribution; store per-page history.
- B3.T1 Create `pages/` and move two pages; keep compatibility routes.
- B4.T1 Dialog component for grant detail; unify downloads via popover.
- B5.T1 Tool registry and Candid fetch tool; guardrail prompts.
- B6.T1 Cache heavy aggregates; sampling for scatter.
- B7.T1 Secrets consolidation; config utilities.
- B8.T1 Pin deps; add dev requirements.
- B9.T1 Add tests; CI workflow.
- B10.T1 Update README and add in-app User Guide.

---

## 7) Success criteria and metrics

- Functional: all original charts work; chat responses feel fast and grounded.
- UX: reduced clicks to common tasks; clearer navigation; consistent styles.
- Perf: interactions < 300ms for filters on sample datasets; large datasets manageable via sampling.
- DevEx: fresh setup in < 5 minutes; CI green; code style consistent.

---

## 8) Open questions / future ideas

- Optional vector search over grant descriptions for semantic filtering.
- Export shareable “insight cards” (image + text) for reports.
- Lightweight user profiles to save views and filters.
- Cloud deployment recipes (Streamlit Community Cloud, Azure, etc.).

---

## 9) Tracking checklist

- [ ] Batch 1 complete
- [ ] Batch 2 complete
- [ ] Batch 3 complete
- [ ] Batch 4 complete
- [ ] Batch 5 complete
- [ ] Batch 6 complete
- [ ] Batch 7 complete
- [ ] Batch 8 complete
- [ ] Batch 9 complete
- [ ] Batch 10 complete
