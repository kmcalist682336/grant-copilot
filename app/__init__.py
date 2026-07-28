"""Web app and prompt lab over the grant-copilot pipeline.

A driver for scripts.chatbot, the same way scripts/chatbot/pipeline_repl.py
is a driver. Owns HTTP, the browser UI, and presentation — and owns no
pipeline logic. The seam is ``orchestrator.answer_query`` plus the
synthesis bundle.

    pipeline_adapter  the ONLY module importing scripts.chatbot
    context           loads the heavy artifacts once
    presentation      hot-reloading config for what the user sees
    recording_llm     transparent proxy that captures every LLM call
    lints             deterministic checks on model output
    postprocess       student-owned cleanup hook
    schemas           the public DTO the frontend binds to
    promptlab         run cache, re-synthesize, prompt variants
    main              FastAPI application

Start it with ``python run_app.py`` or ``docker compose up``.
"""
