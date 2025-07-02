FROM python:3.12

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir

# Copy the entire project
COPY . .

# default streamlit port
EXPOSE 8501

# Run the Streamlit app on all interfaces and default port
CMD ["streamlit", "run", "webapp.py", "--server.port=8501", "--server.address=0.0.0.0"]