# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /modeling

# Copy the requirements.txt to the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /modeling
COPY . .

# Keep the container running by opening a shell
CMD ["tail", "-f", "/dev/null"]
