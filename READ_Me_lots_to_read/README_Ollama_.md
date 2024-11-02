
### Download latest - ollama run llama3.2
- then list it in the Streamlit - ui 
- /usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library


```shell
(base) dhankar@dhankar-1:.../library$ ls -lahtr
total 24K
drwxr-xr-x 2 ollama ollama 4.0K Aug 10 16:31 mistral
drwxr-xr-x 3 ollama ollama 4.0K Aug 10 16:31 ..
drwxr-xr-x 2 ollama ollama 4.0K Aug 10 16:32 nomic-embed-text
drwxr-xr-x 2 ollama ollama 4.0K Oct 26 17:16 llama3.2
drwxr-xr-x 2 ollama ollama 4.0K Oct 26 17:25 llama3.1
drwxr-xr-x 6 ollama ollama 4.0K Oct 26 17:25 .
(base) dhankar@dhankar-1:.../library$ pwd
/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library
(base) dhankar@dhankar-1:.../library$ 

```

#

```shell

^C  Stopping...
(env_llama_idx) dhankar@dhankar-1:~/.../ollama_langchain$ 
(env_llama_idx) dhankar@dhankar-1:~/.../ollama_langchain$ ollama run llama3.2
pulling manifest 
pulling dde5aa3fc5ff... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 2.0 GB                         
pulling 966de95ca8a6... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 1.4 KB                         
pulling fcc5a6bec9da... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 7.7 KB                         
pulling a70ff7e570d9... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 6.0 KB                         
pulling 56bb8bd477a5... 100% ▕██████████████████████████████████████████████████████████████████████████████▏   96 B                         
pulling 34bb5ab01051... 100% ▕██████████████████████████████████████████████████████████████████████████████▏  561 B                         
verifying sha256 digest 
writing manifest 
removing any unused layers 
success 
>>> hello whats the capital of france
Bonjour! The capital of France is Paris.

>>> give me a 10 lines paragraph about GARFIELD the cat 
Here's a paragraph about Garfield:

Garfield is a popular cartoon cat created by Jim Davis. He first appeared on September 11, 1978. Garfield is known for his sarcastic 
personality and love of lasagna. He lives with his owner Jon Arbuckle in Muncie, Indiana. His arch-nemesis is Nermal, a cute grey tabby 
cat. Garfield's lazy nature often gets him into trouble. Despite this, he has a soft spot for Odie, the yellow dog next door. In 2004, 
Garfield became the first cartoon character to have a star on the Hollywood Walk of Fame. He was named the most popular comic strip 
character by the National Cartoonists Society in 1989. Garfield's success led to numerous TV shows, movies, and merchandise.

>>> 
[1]+  Stopped                 ollama run llama3.2
(env_llama_idx) dhankar@dhankar-1:~/.../ollama_langchain$ ollama run llama3.1
pulling manifest 
pulling 8eeb52dfb3bb... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 4.7 GB                         
pulling 948af2743fc7... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 1.5 KB                         
pulling 0ba8f0e314b4... 100% ▕██████████████████████████████████████████████████████████████████████████████▏  12 KB                         
pulling 56bb8bd477a5... 100% ▕██████████████████████████████████████████████████████████████████████████████▏   96 B                         
pulling 1a4c3c319823... 100% ▕██████████████████████████████████████████████████████████████████████████████▏  485 B                         
verifying sha256 digest 
writing manifest 
removing any unused layers 
success 
>>> whats the capital of france
The capital of France is Paris.

>>> give me a 10 lines paragraph about GARFIELD the cat 
Here's a 10-line paragraph about Garfield:

Garfield is a lazy orange tabby cat created by American cartoonist Jim Davis.
He first appeared in comic strips on June 19, 1978, and quickly became popular worldwide.
Garfield is known for his love of food, especially lasagna, which he often eats in large quantities.
Despite his laziness, Garfield is also very intelligent and witty.
He lives with his owner Jon Arbuckle and Jon's dog Odie, who often gets into trouble.
Garfield is a bit cynical and sarcastic, but deep down he has a heart of gold.
He loves to sleep, eat, and watch TV, but will always come to the rescue when needed.
Jon's love life is also a recurring theme in Garfield comics, with Jon trying to win over various women.
Garfield has appeared in numerous comic strips, books, movies, and TV shows over the years.
Despite his many adventures, Garfield remains one of the most beloved cartoon cats of all time.

>>> >>> Send a message (/? for help)
[2]+  Stopped                 ollama run llama3.1
(env_llama_idx) dhankar@dhankar-1:~/.../ollama_langchain$ 


(env_llama_idx) dhankar@dhankar-1:~/.../ollama_langchain$ 
(env_llama_idx) dhankar@dhankar-1:~/.../ollama_langchain$ ollama run llama3.2
pulling manifest 
pulling dde5aa3fc5ff... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 2.0 GB                         
pulling 966de95ca8a6... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 1.4 KB                         
pulling fcc5a6bec9da... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 7.7 KB                         
pulling a70ff7e570d9... 100% ▕██████████████████████████████████████████████████████████████████████████████▏ 6.0 KB                         
pulling 56bb8bd477a5... 100% ▕██████████████████████████████████████████████████████████████████████████████▏   96 B                         
pulling 34bb5ab01051... 100% ▕██████████████████████████████████████████████████████████████████████████████▏  561 B                         
verifying sha256 digest 
writing manifest 
removing any unused layers 
success 
>>> hello whats the capital of france
Bonjour! The capital of France is Paris.

>>> give me a 10 lines paragraph about GARFIELD the cat 
Here's a paragraph about Garfield:

Garfield is a popular cartoon cat created by Jim Davis. He first appeared on September 11, 1978. Garfield is known for his sarcastic 
personality and love of lasagna. He lives with his owner Jon Arbuckle in Muncie, Indiana. His arch-nemesis is Nermal, a cute grey tabby 
cat. Garfield's lazy nature often gets him into trouble. Despite this, he has a soft spot for Odie, the yellow dog next door. In 2004, 
Garfield became the first cartoon character to have a star on the Hollywood Walk of Fame. He was named the most popular comic strip 
character by the National Cartoonists Society in 1989. Garfield's success led to numerous TV shows, movies, and merchandise.

>>> Send a message (/? for help)

```
#
### OLLMA + OpenAI - https://github.com/ollama/ollama/blob/main/docs/openai.md

#

### OLLAMA Models on home Ubuntu - /usr/share/ollama/.ollama/models/blobs
- https://github.com/ollama/ollama/issues/733
- OLLAMA Models on home Ubuntu 
- /usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library
- /usr/share/ollama/.ollama/models/blobs
- OLLAMA Models on WIN OS - `C:\Users\%username%\.ollama\models`
- Supported GPU's from NVIDIA - https://github.com/ollama/ollama/blob/main/docs/gpu.md

#

```
(base) dhankar@dhankar-1:.../library$ ls -lahtr
total 16K
drwxr-xr-x 2 ollama ollama 4.0K Aug 10 16:31 mistral
drwxr-xr-x 3 ollama ollama 4.0K Aug 10 16:31 ..
drwxr-xr-x 2 ollama ollama 4.0K Aug 10 16:32 nomic-embed-text
drwxr-xr-x 4 ollama ollama 4.0K Aug 10 16:32 .
(base) dhankar@dhankar-1:.../library$ 
(base) dhankar@dhankar-1:.../library$ pwd
/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library
```
#

```
OLLAMA Models on home Ubuntu 
(base) dhankar@dhankar-1:.../blobs$ ls -lahtr
total 4.1G
drwxr-xr-x 4 ollama ollama 4.0K Aug 10 16:15 ..
-rw-r--r-- 1 ollama ollama 3.9G Aug 10 16:31 sha256-ff82381e2bea77d91c1b824c7afb83f6fb73e9f7de9dda631bcdbca564aa5435
-rw-r--r-- 1 ollama ollama  12K Aug 10 16:31 sha256-43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1
-rw-r--r-- 1 ollama ollama  801 Aug 10 16:31 sha256-491dfa501e59ed17239711477601bdc7f559de5407fbd4a2a79078b271045621
-rw-r--r-- 1 ollama ollama   30 Aug 10 16:31 sha256-ed11eda7790d05b49395598a42b155812b17e263214292f7b87d15e14003d337
-rw-r--r-- 1 ollama ollama  485 Aug 10 16:31 sha256-42347cd80dc868877d2807869c0e9c90034392b2f1f001cae1563488021e2e19
-rw-r--r-- 1 ollama ollama 262M Aug 10 16:31 sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6
-rw-r--r-- 1 ollama ollama  12K Aug 10 16:31 sha256-c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4
-rw-r--r-- 1 ollama ollama   17 Aug 10 16:31 sha256-ce4a164fc04605703b485251fe9f1a181688ba0eb6badb80cc6335c0de17ca0d
-rw-r--r-- 1 ollama ollama  420 Aug 10 16:32 sha256-31df23ea7daa448f9ccdbbcecce6c14689c8552222b80defd3830707c0139d4f
drwxr-xr-x 2 ollama ollama 4.0K Aug 10 16:32 .
(base) dhankar@dhankar-1:.../blobs$ pwd
/usr/share/ollama/.ollama/models/blobs
(base) dhankar@dhankar-1:.../blobs$ 

```
#

<br>

#
```
Where are Ollama models stored?
Ollama models are stored in the ~/.ollama/models directory on your local machine. This directory contains all the models that you have downloaded or created. The models are stored in a subdirectory named blobs.

When you download a model using the ollama pull command, it is stored in the ~/.ollama/models/manifests/registry.ollama.ai/library/<model family>/latest directory. If you specify a particular version during the pull operation, the model is stored in the ~/.ollama/models/manifests/registry.ollama.ai/library/<model family>/<version> directory.
```
#


















Start for free   →

Trending Topic → LLMOps
Ollama: Easily run LLMs locally
by Stephen M. Walker II, Co-Founder / CEO

What is Ollama?
Ollama is a streamlined tool for running open-source LLMs locally, including Mistral and Llama 2. Ollama bundles model weights, configurations, and datasets into a unified package managed by a Modelfile.

Ollama supports a variety of LLMs including LLaMA-2, uncensored LLaMA, CodeLLaMA, Falcon, Mistral, Vicuna model, WizardCoder, and Wizard uncensored.

Klu OllamaKlu Ollama
Ollama Models
Ollama supports a variety of models, including Llama 2, Code Llama, and others, and it bundles model weights, configuration, and data into a single package, defined by a Modelfile.

The top 5 most popular models on Ollama are:

Model	Description	Pulls	Updated
llama2	The most popular model for general use.	220K	2 weeks ago
mistral	The 7B model released by Mistral AI, updated to version 0.2.	134K	5 days ago
codellama	A large language model that can use text prompts to generate and discuss code.	98K	2 months ago
dolphin-mixtral	An uncensored, fine-tuned model based on the Mixtral MoE that excels at coding tasks.	84K	10 days ago
mistral-openorca	Mistral 7b fine-tuned using the OpenOrca dataset.	57K	3 months ago
llama2-uncensored	Uncensored Llama 2 model by George Sung and Jarrad Hope.	44K	2 months ago
Ollama also supports the creation and use of custom models. You can create a model using a Modelfile, which includes passing the model file, creating various layers, writing the weights, and finally, seeing a success message.

Some of the other models available on Ollama include:

Llama2: Meta's foundational "open source" model.
Mistral/Mixtral: A 7 billion parameter model fine-tuned on top of the Mistral 7B model using the OpenOrca dataset.
Llava: A multimodal model called LLaVA (Large Language and Vision Assistant) which can interpret visual inputs.
CodeLlama: A model trained on both code and natural language in English.
DeepSeek Coder: Trained from scratch on both 87% code and 13% natural language in English.
Meditron: An open-source medical large language model adapted from Llama 2 to the medical domain.
Installation and Setup of Ollama
Download Ollama from the official website.
After downloading, the installation process is straightforward and similar to other software installations. For MacOS and Linux users, you can install Ollama with one command: curl https://ollama.ai/install.sh | sh.
Once installed, Ollama creates an API where it serves the model, allowing users to interact with the model directly from their local machine.
Ollama is compatible with macOS and Linux, with Windows support coming soon. It can be easily installed and used to run various open-source models locally. You can select the model you want to run locally from the Ollama library.

Running Models Using Ollama
Running models using Ollama is a simple process. Users can download and run models using the run command in the terminal. If the model is not installed, Ollama will automatically download it first. For example, to run the Code Llama model, you would use the command ollama run codellama.

Model	RAM	Download Command
Llama 2	16GB	ollama pull llama2
Llama 2 Uncensored	16GB	ollama pull llama2-uncensored
Llama 2 13B	32GB	ollama pull llama2:13b
Orca Mini	8GB	ollama pull orca
Vicuna	16GB	ollama pull vicuna
Nous-Hermes	32GB	ollama pull nous-hermes
Wizard Vicuna Uncensored	32GB	ollama pull wizard-vicuna
Klu Ollama Run ModelKlu Ollama Run Model
Running Ollama Chat UI
Now that you have Ollama running, you can also use Ollama as a local ChatGPT alternative. Download the Ollama WebUI and follow these steps to be running in minutes.

Klu Ollama WebUIKlu Ollama WebUI
Clone the Ollama Web UI repository from GitHub
Follow the README instructions to set up the Web UI.
git clone https://github.com/ollama-webui/ollama-webui.git
Then run the docker run command.

docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v ollama-webui:/app/backend/data --name ollama-webui --restart always ghcr.io/ollama-webui/ollama-webui:main
Or, if you're allergic to Docker, you can use the lite version instead.

git clone https://github.com/ollama-webui/ollama-webui-lite.git
cd ollama-webui-lite
pnpm i && pnpm run dev
And in either case, visit http://localhost:3000 and you're ready for local LLM magic.

Where are Ollama models stored?
Ollama models are stored in the ~/.ollama/models directory on your local machine. This directory contains all the models that you have downloaded or created. The models are stored in a subdirectory named blobs.

When you download a model using the ollama pull command, it is stored in the ~/.ollama/models/manifests/registry.ollama.ai/library/<model family>/latest directory. If you specify a particular version during the pull operation, the model is stored in the ~/.ollama/models/manifests/registry.ollama.ai/library/<model family>/<version> directory.

If you want to import a custom model, you can create a Modelfile with a FROM instruction that specifies the local filepath to the model you want to import. After creating the model in Ollama using the ollama create command, you can run the model using the ollama run command.

Please note that these models can take up a significant amount of disk space. For instance, the 13b llama2 model requires 32GB of storage. Therefore, it's important to manage your storage space effectively, especially if you're working with multiple large models.

Using Ollama with Python
You can also use Ollama with Python. LiteLLM is a Python library that provides a unified interface to interact with various LLMs, including those run by Ollama.

To use Ollama with LiteLLM, you first need to ensure that your Ollama server is running. Then, you can use the litellm.completion function to make requests to the server. Here's an example of how to do this:

from litellm import completion

response = completion(
model="ollama/llama2",
messages=[{ "content": "respond in 20 words. who are you?", "role": "user"}],
api_base="http://localhost:11434"
)

print(response)
In this example, ollama/llama2 is the model being used, and the messages parameter contains the input for the model. The api_base parameter is the address of the Ollama server.

The use case this unlocks is the ability to run LLMs locally, which can be beneficial for several reasons:

Development — Quickly iterate locally without needing to deploy model changes.
Privacy and Security — Running models locally means your data doesn't leave your machine, which can be crucial if you're working with sensitive information.
Cost — Depending on the volume of your usage, running models locally could be more cost-effective than making API calls to a cloud service.
Control — You have more control over the model and can tweak it as needed.
Moreover, LiteLLM's unified interface allows you to switch between different LLM providers easily, which can be useful if you want to compare the performance of different models or if you have specific models that you prefer for certain tasks.

In this example, base_url is the URL where Ollama is serving the model (by default, this is http://localhost:11434), and model is the name of the model you want to use (in this case, llama2).

Additional Features
One of the unique features of Ollama is its support for importing GGUF and GGML file formats in the Modelfile. This means if you have a model that is not in the Ollama library, you can create it, iterate on it, and upload it to the Ollama library to share with others when you are ready.

Available Models
Ollama supports a variety of models, and you can find a list of available models on the Ollama Model Library page.

Ollama supports a variety of large language models. Here are some of the models available on Ollama:

Mistral — The Mistral 7B model released by Mistral AI.
Llama2 — The most popular model for general use.
CodeLlama — A large language model that can use text prompts to generate and discuss code.
Llama2-Uncensored — Uncensored Llama 2 model by George Sung and Jarrad Hope.
Orca-Mini — A general-purpose model ranging from 3 billion parameters to 70 billion, suitable for entry-level hardware.
Vicuna — General use chat model based on Llama and Llama 2 with 2K to 16K context sizes.
Wizard-Vicuna-Uncensored — Wizard Vicuna Uncensored is a 7B, 13B, and 30B parameter model based on Llama 2 uncensored by Eric Hartford.
Phind-CodeLlama — Code generation model based on CodeLlama.
Nous-Hermes — General use models based on Llama and Llama 2 from Nous Research.
Mistral-OpenOrca — Mistral OpenOrca is a 7 billion parameter model, fine-tuned on top of the Mistral 7B model using the OpenOrca dataset.
WizardCoder — Llama based code generation model focused on Python.
Wizard-Math — Model focused on math and logic problems.
Fine-tuned Llama 2 model — To answer medical questions based on an open source medical dataset.
Wizard-Vicuna — Wizard Vicuna is a 13B parameter model based on Llama 2 trained by MelodysDreamj.
Open-Orca-Platypus2 — Merge of the Open Orca OpenChat model and the Garage-bAInd Platypus 2 model. Designed for chat and code generation.
You can find a complete list of available models on the Ollama Model Library page.

Remember to ensure you have adequate RAM for the model you are running. For example, the Code Llama model recommends 8GB of memory for a 7 billion parameter model, 16GB for a 13 billion parameter model, and 32GB for a 34 billion parameter model.

Conclusion
Ollama is a powerful tool for running large language models locally, making it easier for users to leverage the power of LLMs. Whether you're a developer looking to integrate AI into your applications or a researcher exploring the capabilities of LLMs, Ollama provides a user-friendly and flexible platform for running these models on your local machine.

FAQs
Is Ollama open source?
Yes, Ollama is open source. It is a platform that allows you to run large language models, such as Llama 2, locally. Ollama bundles model weights, configuration, and data into a single package, defined by a Modelfile. It optimizes setup and configuration details, including GPU usage.

The source code for Ollama is publicly available on GitHub. In addition to the core platform, there are also open-source projects related to Ollama, such as an open-source chat UI for Ollama.

Ollama supports a list of open-source models available on its library. These models are trained on a wide variety of data and can be downloaded and used with the Ollama platform.

To use Ollama, you can download it from the official website, and it is available for macOS and Linux, with Windows support coming soon. There are also tutorials available online that guide you on how to use Ollama to build open-source versions of various applications.

What does Ollama do?
Ollama is a tool that allows you to run open-source large language models (LLMs) locally on your machine. It supports a variety of models, including Llama 2, Code Llama, and others. It bundles model weights, configuration, and data into a single package, defined by a Modelfile.

Ollama is an extensible platform that enables the creation, import, and use of custom or pre-existing language models for a variety of applications, including chatbots, summarization tools, and creative writing aids. It prioritizes privacy and is free to use, with seamless integration capabilities for macOS and Linux users, and upcoming support for Windows.

The platform streamlines the deployment of language models on local machines, offering users control and ease of use. Ollama's library (ollama.ai/library) provides access to open-source models such as Mistral, Llama 2, and Code Llama, among others.

System requirements for running models vary; a minimum of 8 GB of RAM is needed for 3B parameter models, 16 GB for 7B, and 32 GB for 13B models. Additionally, Ollama can serve models via a REST API for real-time interactions.

Use cases for Ollama are diverse, ranging from LLM-powered web applications to integration with local note-taking tools like Obsidian. In summary, Ollama offers a versatile and user-friendly environment for leveraging the capabilities of large language models locally for researchers, developers, and AI enthusiasts.

More terms
What is Human Intelligence?
Human Intelligence refers to the mental quality that consists of the abilities to learn from experience, adapt to new situations, understand and handle abstract concepts, and use knowledge to manipulate one's environment. It is a complex ability influenced by various factors, including genetics, environment, culture, and education.

Read more
What is Software 2.0?
Software 2.0 refers to the new generation of software that is written in the language of machine learning and artificial intelligence. Unlike traditional software that is explicitly programmed, Software 2.0 learns from data and improves over time. It can perform complex tasks such as natural language processing, pattern recognition, and prediction, which are difficult or impossible for traditional software. The capabilities of Software 2.0 extend beyond simple data entry and can include advanced tasks like facial recognition and understanding natural language.

Read more
It's time to build
Collaborate with your team on reliable Generative AI features.
Want expert guidance? Book a 1:1 onboarding session from your dashboard.

Start for free   →
LLMOps
Getting Started with LLMOps
LLM Evaluation
Azure OpenAI
MT Bench Eval
Guides
Prompt Engineering
LLM App Optimization
Open Source LLMs
GPT 3.5 vs. Gemini Pro
LLMs
LLM Leaderboard
Introduction to LLMs
Reinforcement Learning from Human Feedback
Reinforcement Learning from AI Feedback
Docs
LLM Leaderboard
Blog
Glossary
Releases
Privacy
MSA
Help
K-human Likeness Utility © Klu, Inc.

