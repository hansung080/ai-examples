# AI Agents
Various AI agents

## Runtime Environment
### Create Virtual Environment
To create a virtual environment and install dependent packages, run the following commands in the project root directory.
```sh
$ python3.12 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip3 install -r requirements.txt
(.venv) $ deactivate
```

### Create Configuration File
Create the `.env` file with the following content in the project root directory.
```text
OPENAI_API_KEY=<openai-api-key>
GPT_MODEL=gpt-4o-mini
```

## Usage
### Run General Application
To run a general application, run the following commands on the system environment.
```sh
$ cd <python-file-directory> 
$ ./<python-file> [arguments]
```

### Run Streamlit Application
To run a Streamlit application, run the following commands on the virtual environment. 
```sh
$ source .venv/bin/activate
(.venv) $ streamlit run <python-file> [arguments]
```

## Agents
- gpt: gpt_basic, one_shot, few_shot, single_turn, multi_turn, streamlit_basic 
