## Robot Imitaion Learning


### Building Docker image
Running using docker:

1. In the project folder, cd into the robot_imitaion_learning
2. To build docker image, run command: docker build -t imitation_learning:v1 .
3. To enable running GUI application, run command : xhost +local:docker
4. To run docker image in interactive mode,(be in workspace directory) run command: docker run -it --rm --name=enpm690_final_project --net=host --pid=host --privileged --env="DISPLAY=$DISPLAY" imitation_learning:v1 

### Running Simulation

1. Open project folder,cd into ros_ws and from here open two terminals.
2. (Skip this if not using docker) In any terminal, run command : xhost +local:docker
3. (Skip this if not using docker) In the first terminal, run the above mentioned command to run the docker image interactively. In other terminal, run : docker ps to view running images. run command docker exec -it [container_id] bash 

