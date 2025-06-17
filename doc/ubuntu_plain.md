## Workflow

![Workflow](../galleries/workflow.jpg)

##  Installation

### Step1
Then we start a container mapped to an external volume to store the projects permanently:

    docker run 	-d --name flowise \
			    -v <your local folder>/root/.flowise \
			    -p 8000:3000 flowise
After one or two minutes, you can access flowise in

    http://localhost:8000/


### Step2:
Open the sidetab Chatflows or Agentflows and add a new one:

 - Import one model as JSON from **src/FlowiseModel**. 
 - Two models are available: 
	 - *GithubFlowiseOpenAI Chatflow.json* or *Visual Chatflow Deepseek_Ollama.json*
 - In the OpenAI version, add your **OpenAI API Key** in the chatModel.
 - In the local Ollama version, make sure, that:
> your Ollama Server is started with:

    OLLAMA_HOST=0.0.0.0:11434 ollama serve
> and your ChatOllama in your imported Flowise model is set to: 

    http://"your local PC IPaddress":11434

> Select the model with the name, eg. 

`deepseek-r1:14b`


> Written with [StackEdit](https://stackedit.io/).
