#base image
FROM python:3.13-slim

# workdir
WORKDIR /app

#copy requirements file
COPY requirements.txt .

#run
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

#copy rest of the project files
COPY . .

#port
EXPOSE 8501

#command
CMD [ "streamlit","run","./UI/app.py","--server.address=0.0.0.0", "--server.port=8501" ] 