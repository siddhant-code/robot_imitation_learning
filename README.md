## Robot Imitaion Learning


### Building Docker image
Running using docker:

1. In the project folder, cd into the robot_imitaion_learning
2. To build docker image, run command: docker build -t imitation_learning:v1 .
3. To enable running GUI application, run command : xhost +local:docker
4. To run docker image in interactive mode,(be in workspace directory) run command: docker run -it --rm --name=enpm690_final_project --net=host --pid=host --privileged --env="DISPLAY=$DISPLAY" imitation_learning:v1 

### Running Simulation

1. Make sure you are in /robot_imitation_learning folder to follow the steps ahead.
2. To do imitation learning using behaviour cloning, use command : python3 imitation_learning_train.py
3. To visualize final trained model,use command : python3 imitation_learning_visualize.py
4. To train model usin GAIL, use command : python3 gail_learning_train.py
5. To visualize final trained gail model,use command : python3 gail_learning_visualize.py

