FROM python:3.12-slim

RUN python -m pip install --no-cache-dir playwright==1.55.0 pillow==12.3.0 httpx imageio-ffmpeg pytest \
    && python -m playwright install --with-deps chromium

WORKDIR /work
ENV PYTHONPATH=/work
