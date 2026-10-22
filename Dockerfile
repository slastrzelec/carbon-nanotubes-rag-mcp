# MCP server over streamable HTTP. Holds no LLM key; needs RAG_MCP_HTTP_TOKEN (>= 32 chars) at run time.
#   docker build -t carbon-nanotubes-rag-mcp .
#   docker run --rm -p 127.0.0.1:8765:8765 -e RAG_MCP_HTTP_TOKEN=... \
#     -e RAG_MCP_BACKEND_URL=http://host.docker.internal:8000 -e RAG_MCP_ALLOWED_HOSTS=host.docker.internal \
#     carbon-nanotubes-rag-mcp
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . \
    && useradd --create-home --uid 10001 app

USER app
# Inside the container the server must listen on all interfaces; publish the port on 127.0.0.1 only.
ENV RAG_MCP_HTTP_HOST=0.0.0.0 \
    RAG_MCP_HTTP_PORT=8765
EXPOSE 8765
ENTRYPOINT ["rag-mcp", "--transport", "http"]
