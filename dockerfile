FROM python:3.10

RUN apt-get update && apt-get install -y cmake 

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /robot_imitation_learning

COPY ./ ./robot_imitation_learning/

RUN cd /robot_imitation_learning/

RUN pip install -r ./robot_imitation_learning/requirements.txt

CMD ["bash"]