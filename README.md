### CamelAI Laboratories (Lab 1 & 2)
The repository contains the code that was presented as part of the Agent Systems and Applications CamelAI laboratories at WUT.

### Prerequisites
1. Python (at least 3.10 BUT smaller than 3.13) -> https://www.python.org/
2. [CamelAI]([https://mesa.readthedocs.io/latest/index.html](https://www.camel-ai.org/)) (with all libraries) -> `pip install camel-ai[rec]`

### Model set-up
CamelAI supports connection to both (1) external APIs and (2) local models.

#### Setting up Ollama
1. Download Ollama from the [Ollama website](https://ollama.com/download)
2. Pull the model (e.g. llama3): `ollama pull llama3`
3. From the class _DefaultModel_ (e.g. in _laboratory1/model_) refer to the method `create_local_model` in order to initialize local model

#### Setting up OpenAI API
1. Generate your individual API key in [API-KEYS section](https://platform.openai.com/api-keys) (IMPORTANT! You will need tokens to make API requests, so be aware of pricing!)
2. Paste your key into corresponding config file: _laboratory1/config/config.py_ or _laboratory1/config/config.py_ and set up the OPENAI_API_KEY environmental variable 
(technically, it is only needed to follow one of these options, but for the purpose of running all presented examples, it is adviced to do both)
3. From the class _DefaultModel_ (e.g. in _laboratory1/model_) refer to the methods `create_openai_model` and  `create_custom_openai_model` in order to initialize OpenAI model
