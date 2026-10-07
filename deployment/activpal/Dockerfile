FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY deployment/activpal/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the WEALTH package (from local files)
COPY pyproject.toml .
COPY src ./src

RUN pip install --no-cache-dir .

# Copy the inference application
COPY deployment/activpal/inference.py .

# Create input/output directories
RUN mkdir -p /app/input /app/output

CMD ["python", "inference.py"]
