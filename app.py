"""Chainlit chat UI for a remote Ollama server.

Run with:
    pip install chainlit ollama
    chainlit run app.py -w

Override the server address with the OLLAMA_HOST env var if needed.
"""

import base64
import os
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select, Slider
from ollama import AsyncClient, ResponseError

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.1.153:11434")

client = AsyncClient(host=OLLAMA_HOST)


def _build_user_message(message: cl.Message) -> dict:
    """Merge plain text with uploaded files/images before sending to Ollama."""
    prompt = (message.content or "").strip()
    text_parts = []
    images = []
    seen_paths = set()

    for raw_item in list(getattr(message, "elements", []) or []) + list(
        getattr(message, "attachments", []) or []
    ):
        if raw_item is None:
            continue

        if isinstance(raw_item, dict):
            path = raw_item.get("path") or raw_item.get("url") or raw_item.get("name")
            mime = raw_item.get("mime") or raw_item.get("type") or ""
            name = raw_item.get("name") or raw_item.get("filename") or "uploaded file"
        else:
            path = getattr(raw_item, "path", None) or getattr(raw_item, "url", None)
            mime = getattr(raw_item, "mime", None) or getattr(raw_item, "type", None) or ""
            name = getattr(raw_item, "name", None) or getattr(raw_item, "filename", None) or "uploaded file"

        if not path or not os.path.exists(path):
            continue

        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)

        lower_path = str(path).lower()
        is_image = mime.lower().startswith("image/") or lower_path.endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
        )

        if is_image:
            try:
                with open(path, "rb") as image_file:
                    images.append(base64.b64encode(image_file.read()).decode("utf-8"))
                text_parts.append(f"Image attachment: {name}")
            except OSError:
                text_parts.append(f"Image attachment: {name} (unreadable)")
            continue

        try:
            file_text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        if file_text.strip():
            text_parts.append(f"Attached file: {name}\n{file_text.strip()}")

    if text_parts:
        combined = "\n\n".join(part for part in text_parts if part)
        if prompt:
            prompt = f"{prompt}\n\n{combined}"
        else:
            prompt = combined

    payload = {"role": "user", "content": prompt or "Please analyze the uploaded file(s)."}
    if images:
        payload["images"] = images
    return payload


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
    user_message = _build_user_message(message)
    history.append(user_message)

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
