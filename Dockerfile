# Use a lightweight version of Python
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your other files (app.py, agents folder, etc.) into the container
COPY . .

# Expose the port (assuming this is a Streamlit app)
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
