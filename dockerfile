FROM python:3.10

RUN apt-get update && apt-get install -y cmake 

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /robot_imitation_learning

COPY ./ ./

RUN pip install -r ./requirements.txt

CMD ["bash"]