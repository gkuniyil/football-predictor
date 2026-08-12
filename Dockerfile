# Base image: a lightweight Python 3.9 environment (matches your local venv version)
FROM python:3.9-slim

# All subsequent commands run from /app inside the container
WORKDIR /app

# Copy just requirements.txt first (not the whole project yet) --
# this lets Docker cache the pip install step, so rebuilding after a code
# change doesn't re-download every package from scratch every time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual project code and data
COPY src/ src/
COPY data/ data/

# Tell the container to run uvicorn when it starts.
# --host 0.0.0.0 is required (not 127.0.0.1) so the API is reachable
# from OUTSIDE the container, not just from within it.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]