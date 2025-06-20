## Workflow

Two docker container are started: 

 - **Flowise** Container.
 - **CAM** Container with data processing and data posting to Flowise.
 
 The difference to the plain setup is, the CAM is piping directly to Flowise instead of using ECAL as communication layer. Also we are using a Flask server to show the video frames and the results outside the docker container.
  
##  Installation

### Step 1: (only once, if dockerfile not modificated)
Change into the Docker folder ./src/Docker and build the container:

    docker build -t flowiseapp .

### Step 2: (only once)
Allow docker to access your X server

    xhost +local:root

Create a docker network

    docker network create flowise-shared-net

### Step 3:
Create the Flowise container with access to the docker network:

    docker run -d --name flowise -v /<your flowise storage location>/.flowise/:/root/.flowise --network flowise-shared-net -p 8000:3000 flowise

or if already started

    docker restart flowise

You can check the configuration of the container with 

    docker inspect flowise
It should contain:

 - "NetworkMode": "flowise-shared-net"
 - "PortBindings": ... "HostPort": "8000"

### Step 4: Workcontainer Setup

We setup our workontainer with video streaming and flowise communication:

    docker run -d --name flowiseapp --gpus all --privileged --device=/dev/video0 -p 5555:5555 --network flowise-shared-net flowiseapp
or if already started

    docker restart flowiseapp


You can check the configuration of the container with 

    docker inspect flowiseapp
It should contain:

 - "NetworkMode": "flowise-shared-net"
 - "PortBindings": ... "HostPort": "5555"
 -  "Path": "python3",
        "Args": [
            "/app/cam2flowise.py"


### Step5: Visualisation

In the same folder start:

    python webcam_stream_server.py

### Result:
[Flaskserver](../galleries/docker_linux.png)


> Written with [StackEdit](https://stackedit.io/).
