# AI Examples
Various AI code examples

## Runtime Environment
### Create Virtual Environment
To create a virtual environment and install dependent packages, run the following commands in the project root directory.
```sh
$ python3.13 -m venv .venv
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
### Run Standalone Application
To run a standalone application, run the following commands on the system environment.
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

## AI Examples
- Do it! AI Agent Development with LLM (aadl)
  - gpt_basic, one_shot, few_shot, single_turn, multi_turn, streamlit_basic
- Essential Math for Data Science (emds)
  - nn_for_bin_class 
