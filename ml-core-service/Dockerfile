# Stage 1: Builder Stage
# This stage installs dependencies and gathers source code.
FROM python:3.11-slim-bullseye AS builder

# Set environment variables to prevent python from writing pyc files and to buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv

# Make the venv's binaries accessible
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Stage 2: Runner Stage
# This stage creates the final, lean production image.
FROM python:3.11-slim-bullseye AS runner

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code from the builder stage
COPY --from=builder /app .

# Create and switch to a non-root user for better security
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port the app will run on
EXPOSE 8000

# Make the venv's binaries accessible
ENV PATH="/opt/venv/bin:$PATH"

# Run the Uvicorn server, pointing to the 'app' instance in 'server.py'
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers=5"]

# Trigger build 3
