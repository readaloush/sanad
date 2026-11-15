# Sanad search and answer service. No PyTorch in the image: ONNX Runtime, tokenizers and numpy only.
# The language model runs elsewhere behind an OpenAI-compatible API (vLLM, llama.cpp, Ollama).
#   python -m sanad.build_index --out artifacts
#   docker build -t sanad .
#   docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" -e SANAD_LLM_URL=http://host:8001 sanad
FROM python:3.12-slim
WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt
COPY sanad/__init__.py sanad/prompts.py sanad/serve.py sanad/
COPY service service
ENV SANAD_ARTIFACTS=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
