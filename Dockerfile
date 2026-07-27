FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install torch torchvision --index-url ${TORCH_INDEX_URL} \
    && python -m pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "train.py"]
CMD ["--help"]
