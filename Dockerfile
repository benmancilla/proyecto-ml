# Multi-stage Dockerfile for ML Project with Kedro, DVC, and Airflow
# Optimized for reproducibility and portability

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install kedro~=0.19.11 \
                dvc \
                apache-airflow>=2.8.0 \
                kedro-viz

# Copy project files
COPY . .

# Install project in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/01_raw \
             data/02_intermediate \
             data/03_primary \
             data/05_model_input \
             data/06_models \
             data/07_model_output \
             data/08_reports

# Expose ports
# 8080: Airflow webserver
# 4141: Kedro-viz
EXPOSE 8080 4141

# Default command
CMD ["kedro", "run"]


# Development stage (optional)
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov black flake8 ipython jupyterlab

# Set up Jupyter
RUN jupyter lab --generate-config

CMD ["bash"]


# Production stage
FROM base as production

# Set user for security (non-root)
RUN useradd -m -u 1000 kedro && \
    chown -R kedro:kedro /app

USER kedro

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["kedro", "run"]
