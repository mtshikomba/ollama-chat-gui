"""Chainlit chat UI for a remote Ollama server.

Run with:
    pip install chainlit ollama
    chainlit run app.py -w

Override the server address with the OLLAMA_HOST env var if needed.
"""

import os

import chainlit as cl
from chainlit.input_widget import Select, Slider
from ollama import AsyncClient, ResponseError

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.1.153:11434")

client = AsyncClient(host=OLLAMA_HOST)


async def list_models() -> list[str]:
    """Return the model names available on the remote server."""
    data = await client.list()
    models = []
    for m in data.get("models", []):
        name = m.get("model") or m.get("name")
        if name:
            models.append(name)
    return sorted(models)


@cl.on_chat_start
async def on_chat_start():
    try:
        models = await list_models()
    except Exception as exc:  # noqa: BLE001 - surface any connection problem to the user
        await cl.Message(
            content=(
                f"Could not reach Ollama at `{OLLAMA_HOST}`.\n\n"
                f"```\n{exc}\n```\n\n"
                "Check that the server is running and reachable, then reload."
            )
        ).send()
        return

    if not models:
        await cl.Message(
            content=(
                f"Connected to `{OLLAMA_HOST}`, but no models are installed.\n\n"
                "Pull one on the server, e.g. `ollama pull llama3.1`, then reload."
            )
        ).send()
        return

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=models,
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.8,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="num_ctx",
                label="Context window (tokens)",
                initial=4096,
                min=1024,
                max=32768,
                step=1024,
            ),
        ]
    ).send()

    cl.user_session.set("settings", settings)
    cl.user_session.set("history", [])

    await cl.Message(
        content=(
            f"Connected to Ollama at `{OLLAMA_HOST}`.\n\n"
            f"**{len(models)} model(s) available:** {', '.join(f'`{m}`' for m in models)}\n\n"
            f"Using `{settings['model']}`. Change it any time via the settings panel."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("settings", settings)


@cl.on_message
async def on_message(message: cl.Message):
    settings = cl.user_session.get("settings") or {}
    model = settings.get("model")
    if not model:
        await cl.Message(content="No model selected. Reload the chat to connect.").send()
        return

    history = cl.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})

    options = {
        "temperature": settings.get("temperature", 0.8),
        "num_ctx": int(settings.get("num_ctx", 4096)),
    }

    reply = cl.Message(content="")
    await reply.send()

    full = ""
    try:
        stream = await client.chat(
            model=model,
            messages=history,
            options=options,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                full += token
                await reply.stream_token(token)
    except ResponseError as exc:
        await reply.stream_token(f"\n\n**Ollama error:** {exc}")
        reply.content = reply.content or f"Ollama error: {exc}"
        await reply.update()
        return
    except Exception as exc:  # noqa: BLE001
        await reply.stream_token(f"\n\n**Connection error:** {exc}")
        await reply.update()
        return

    history.append({"role": "assistant", "content": full})
    cl.user_session.set("history", history)
    await reply.update()
