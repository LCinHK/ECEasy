# Server Split Notes

`eceasy_local_server.py` is now a compatibility launcher only.
Core logic moved to `eceasy_server/`.

## New Structure

- `eceasy_server/app.py` - FastAPI app, lifespan, routes, middleware, static mounts
- `eceasy_server/config.py` - .env loading and runtime settings/constants
- `eceasy_server/schemas.py` - request/response Pydantic models
- `eceasy_server/llm.py` - LLM runtime resolution and provider/client selection
- `eceasy_server/retrieval.py` - RAG + web retrieval and related-question generation
- `eceasy_server/streaming.py` - stream pipeline and cache write path
- `eceasy_server/services.py` - compatibility facade re-exporting split service functions
- `eceasy_server/__init__.py` - package exports

## Compatibility

You can still run the server with the same command:

```python
python eceasy_local_server.py
```

`eceasy_local_server.py` re-exports `app` and uses `HOST`/`PORT` from the split config.

## Validation Performed

- Python compile check for all split files
- Import check for compatibility entrypoint (`hasattr(eceasy_local_server, "app")`)

