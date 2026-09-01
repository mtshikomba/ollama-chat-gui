"""Native desktop chat client for a remote Ollama server.

Run from source:
    pip install PySide6 requests
    python desktop_app.py

Build a single-file executable:
    pip install pyinstaller
    pyinstaller --onefile --windowed --name ollama-chat desktop_app.py

Override the server address with the OLLAMA_HOST env var.
"""

import json
import os
import sys

import requests
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.1.153:11434").rstrip("/")
REQUEST_TIMEOUT = 300

APP_STYLESHEET = """
QApplication {
    background-color: #070d18;
    color: #e6edf8;
}

QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0b1220, stop:1 #070d18);
    color: #e6edf8;
}

QWidget {
    color: #e6edf8;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}

QLabel {
    color: #d9e5f5;
}

QComboBox,
QDoubleSpinBox,
QSpinBox,
QPushButton,
QPlainTextEdit,
QTextEdit {
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(15, 23, 42, 0.8);
    color: #e6edf8;
    border-radius: 12px;
    padding: 8px 10px;
}

QComboBox,
QDoubleSpinBox,
QSpinBox {
    min-height: 34px;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7dd3fc, stop:1 #5b7cff);
    border: none;
    color: #061120;
    font-weight: 700;
    padding: 9px 14px;
    min-height: 34px;
}

QPushButton:disabled {
    background: rgba(148, 163, 184, 0.14);
    color: rgba(230, 237, 248, 0.6);
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #9ae6ff, stop:1 #6d8cff);
}

QTextEdit,
QPlainTextEdit {
    border-radius: 16px;
    padding: 14px 16px;
    selection-background-color: rgba(125, 211, 252, 0.4);
    selection-color: #ffffff;
}

QStatusBar {
    background: rgba(15, 23, 42, 0.8);
    color: #d9e5f5;
}

QScrollBar:vertical {
    background: rgba(15, 23, 42, 0.5);
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: rgba(125, 211, 252, 0.35);
    border-radius: 5px;
}
"""


def escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_markdown_block(text: str) -> str:
    value = escape_html(text).replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\n", "<br>")

    lines = value.split("<br>")
    html_parts = []
    in_code = False
    code_lines = []
    in_list = False
    list_items = []

    def flush_list():
        if list_items:
            html_parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>")
            list_items.clear()

    def flush_code():
        nonlocal in_code, code_lines
        if in_code:
            html_parts.append("<pre><code>" + "".join(code_lines) + "</code></pre>")
            code_lines = []
            in_code = False

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("```"):
            flush_list()
            if in_code:
                flush_code()
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(raw_line + "\n")
            continue

        if line.startswith("- ") or line.startswith("* "):
            flush_list()
            list_items.append(line[2:])
            in_list = True
            continue

        if in_list and not (line.startswith("- ") or line.startswith("* ")):
            flush_list()
            in_list = False

        if line.startswith("> "):
            html_parts.append(f"<blockquote>{line[2:]}</blockquote>")
            continue

        if line.startswith("# "):
            html_parts.append(f"<h3>{line[2:]}</h3>")
            continue

        if line.startswith("## "):
            html_parts.append(f"<h4>{line[3:]}</h4>")
            continue

        if line:
            html_parts.append(f"<p>{line}</p>")

    flush_list()
    flush_code()
    return "".join(html_parts) or "<p></p>"


class ModelsWorker(QThread):
    """Fetch the list of installed models without blocking the UI."""

    ok = Signal(list)
    failed = Signal(str)

    def run(self):
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=15)
            resp.raise_for_status()
            names = []
            for m in resp.json().get("models", []):
                name = m.get("model") or m.get("name")
                if name:
                    names.append(name)
            self.ok.emit(sorted(names))
        except Exception as exc:  # noqa: BLE001 - report any failure to the user
            self.failed.emit(str(exc))


class ChatWorker(QThread):
    """Stream a chat completion from Ollama."""

    token = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, model, messages, options):
        super().__init__()
        self.model = model
        self.messages = messages
        self.options = options
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        parts = []
        try:
            with requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.messages,
                    "options": self.options,
                    "stream": True,
                },
                stream=True,
                timeout=REQUEST_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if self._stop:
                        break
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        self.failed.emit(data["error"])
                        return
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        parts.append(chunk)
                        self.token.emit(chunk)
                    if data.get("done"):
                        break
            self.done.emit("".join(parts))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MessageInput(QPlainTextEdit):
    """Multi-line input that sends on Enter (Shift+Enter for a newline)."""

    submit = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            self.submit.emit()
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Ollama Chat — {OLLAMA_HOST}")
        self.resize(900, 700)

        self.messages = []
        self.chat_worker = None
        self.models_worker = None
        self._assistant_open = False

        QApplication.instance().setStyleSheet(APP_STYLESHEET)
        self._build_ui()
        self.refresh_models()

    # ---------- UI ----------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Model:"))
        self.model_box = QComboBox()
        self.model_box.setMinimumWidth(220)
        bar.addWidget(self.model_box)

        self.reload_btn = QPushButton("Reload")
        self.reload_btn.clicked.connect(self.refresh_models)
        bar.addWidget(self.reload_btn)

        bar.addStretch(1)

        bar.addWidget(QLabel("Temp:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.8)
        bar.addWidget(self.temp_spin)

        bar.addWidget(QLabel("Context:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(1024, 131072)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setValue(4096)
        bar.addWidget(self.ctx_spin)

        self.new_btn = QPushButton("New chat")
        self.new_btn.clicked.connect(self.new_chat)
        bar.addWidget(self.new_btn)

        root.addLayout(bar)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        root.addWidget(self.transcript, 1)

        bottom = QHBoxLayout()
        self.input = MessageInput()
        self.input.setPlaceholderText("Message… (Enter to send, Shift+Enter for newline)")
        self.input.setFixedHeight(80)
        self.input.submit.connect(self.send)
        bottom.addWidget(self.input, 1)

        buttons = QVBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send)
        buttons.addWidget(self.send_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        buttons.addWidget(self.stop_btn)
        bottom.addLayout(buttons)

        root.addLayout(bottom)

        self.status = self.statusBar()
        self.status.showMessage(f"Connecting to {OLLAMA_HOST}…")

    # ---------- transcript helpers ----------

    def _append_html(self, html):
        self.transcript.append(html)

    def _append_text_at_end(self, text):
        cur = self.transcript.textCursor()
        cur.movePosition(QTextCursor.End)
        self.transcript.setTextCursor(cur)
        self.transcript.insertHtml(render_markdown_block(text))
        self.transcript.ensureCursorVisible()

    def _append_user_message(self, text):
        self._append_html(f"<div style='margin: 12px 0 8px;'><strong style='color:#e6edf8;'>You</strong></div>")
        self._append_text_at_end(text)

    def _append_assistant_message(self, text, model):
        self._append_html(f"<div style='margin: 12px 0 8px;'><strong style='color:#7dd3fc;'>{model}</strong></div>")
        self._append_text_at_end(text)

    # ---------- models ----------

    def refresh_models(self):
        self.reload_btn.setEnabled(False)
        self.status.showMessage(f"Fetching models from {OLLAMA_HOST}…")
        self.models_worker = ModelsWorker()
        self.models_worker.ok.connect(self._models_ok)
        self.models_worker.failed.connect(self._models_failed)
        self.models_worker.start()

    def _models_ok(self, names):
        self.reload_btn.setEnabled(True)
        current = self.model_box.currentText()
        self.model_box.clear()
        self.model_box.addItems(names)
        if current in names:
            self.model_box.setCurrentText(current)
        if names:
            self.status.showMessage(f"Connected to {OLLAMA_HOST} — {len(names)} model(s)")
        else:
            self.status.showMessage(
                f"Connected to {OLLAMA_HOST}, but no models installed (ollama pull …)"
            )

    def _models_failed(self, err):
        self.reload_btn.setEnabled(True)
        self.status.showMessage(f"Could not reach {OLLAMA_HOST}")
        QMessageBox.warning(
            self,
            "Connection failed",
            f"Could not reach Ollama at {OLLAMA_HOST}.\n\n{err}\n\n"
            "Make sure the server is running and listening on the network "
            "(OLLAMA_HOST=0.0.0.0:11434 on that machine).",
        )

    # ---------- chat ----------

    def new_chat(self):
        if self.chat_worker and self.chat_worker.isRunning():
            return
        self.messages = []
        self.transcript.clear()
        self.status.showMessage("New conversation")

    def send(self):
        if self.chat_worker and self.chat_worker.isRunning():
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        model = self.model_box.currentText()
        if not model:
            QMessageBox.information(self, "No model", "No model selected.")
            return

        self.input.clear()
        self.messages.append({"role": "user", "content": text})
        self._append_user_message(text)
        self._append_assistant_message("", model)
        self._assistant_open = True

        options = {
            "temperature": self.temp_spin.value(),
            "num_ctx": self.ctx_spin.value(),
        }

        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.showMessage(f"Generating with {model}…")

        self.chat_worker = ChatWorker(model, list(self.messages), options)
        self.chat_worker.token.connect(self._on_token)
        self.chat_worker.done.connect(self._on_done)
        self.chat_worker.failed.connect(self._on_failed)
        self.chat_worker.start()

    def _on_token(self, tok):
        self._append_text_at_end(tok)

    def _on_done(self, full):
        if full:
            self.messages.append({"role": "assistant", "content": full})
        self._append_text_at_end("\n")
        self._assistant_open = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.showMessage("Ready")

    def _on_failed(self, err):
        self._append_text_at_end(f"\n[error: {err}]\n\n")
        self._assistant_open = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.showMessage(f"Error: {err}")

    def stop_generation(self):
        if self.chat_worker and self.chat_worker.isRunning():
            self.chat_worker.stop()
            self.status.showMessage("Stopping…")

    def closeEvent(self, event):
        if self.chat_worker and self.chat_worker.isRunning():
            self.chat_worker.stop()
            self.chat_worker.wait(2000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
