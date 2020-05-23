FROM pytorch/pytorch:1.5.0-py3

COPY requirements.txt requirements.txt 

RUN pip3 install -r requirements.txt

COPY . . 
# expose port to run the server
EXPOSE 5000
CMD ["python3", "server.py"]
