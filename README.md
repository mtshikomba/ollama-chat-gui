# Ollama Chat GUI

A lightweight local chat UI for running Ollama models in a browser or desktop wrapper.

## What this is

This project connects to a running Ollama instance and lets you:

- select an installed model
- send prompts to that model
- stream the response back in real time
- adjust temperature and context window settings
- optionally upload text/image files for model analysis

## Requirements

- Python 3.9+
- Ollama installed and running locally
- A model pulled in Ollama, for example:

```bash
ollama pull llama3.1
```

## Local setup

Create and activate a virtual environment in the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install chainlit ollama
```

Then start the app:

```bash
chainlit run app.py -w
```

If your Ollama server is not on the default host, set it before starting:

```bash
export OLLAMA_HOST=http://localhost:11434
```

## Notes

- The app expects Ollama to be running and reachable.
- If no models are installed, pull one first with `ollama pull ...`.
- The desktop app is also included, but the browser app is the main workflow for local use.
