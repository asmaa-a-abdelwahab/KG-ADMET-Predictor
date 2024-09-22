# The Dockerfile for the Streamlit app.

# The base image is the latest Python 3.9 image from Docker Hub.
FROM python:3.9.5-slim-buster

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the latest version of pip
RUN pip install -U pip

# Install the dependencies from requirements.txt without caching
RUN pip install --no-cache-dir -r requirements.txt

# Install nano text editor to be able to edit the code in the container
RUN apt-get update && apt-get install nano -y

# Set the default command for the container to run the Streamlit app
CMD streamlit run app.py
