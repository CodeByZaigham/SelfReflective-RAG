#base image
FROM python:3.13

# workdir
WORKDIR /app

#copy 
COPY . .

#run
RUN pip install -r requirements.txt

#port
EXPOSE 8501

#command
CMD [ "streamlit","run","./UI/app.py","--server.address=0.0.0.0", "--server.port=8501" ] 