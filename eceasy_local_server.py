"""
Compatibility entrypoint for ECEasy local server.

The server implementation has been split into modules under `eceasy_server/`.
This file intentionally stays as the stable launcher used by existing scripts.
"""

from eceasy_server.app import app
from eceasy_server.config import HOST, PORT


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
