FROM python:3.11-slim

LABEL maintainer="su-memory team"
LABEL description="su-memory v2.0 — HotpotQA Distractor Submission (retrieval + QA extraction)"
LABEL version="2.0"

WORKDIR /workspace

# Install Python dependencies (downloaded during build, not evaluation)
RUN pip install --no-cache-dir transformers torch

# Pre-download the QA model during build
RUN python3 -c "from transformers import pipeline; pipeline('question-answering', model='distilbert-base-cased-distilled-squad')"

# Copy submission script
COPY run.py /workspace/run.py

ENTRYPOINT ["python3", "/workspace/run.py"]
CMD ["input.json", "pred.json"]
