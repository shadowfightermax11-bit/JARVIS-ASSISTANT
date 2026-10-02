import os
import re
import json
import time
import tempfile
import webbrowser
import subprocess
import asyncio
import threading
import ctypes
import shutil
import urllib.request
import urllib.error
from urllib.parse import urlparse, parse_qs, quote_plus
from http.server import BaseHTTPRequestHandler, HTTPServer

import edge_tts
import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

from scipy.signal import resample_poly
from openwakeword.model import Model
from google import genai

import local_responses


# ============================================================
# J.A.R.V.I.S. SETTINGS
# ============================================================

APP_NAME = "JARVIS Companion"
APP_VERSION = "3.0.0"

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 8765

VOICE = "en-US-AndrewNeural"

WAKE_WORD = "hey_jarvis"
WAKE_THRESHOLD = 0.50

MAX_COMMAND_TIME = 7.0
END_SILENCE = 1.0
MIN_SPEECH_TIME = 0.25
COOLDOWN = 1.5

MAX_HISTORY = 40


# ============================================================
# JARVIS USER DATA DIRECTORY
# ============================================================
#
# IMPORTANT:
# The installed program should not write user data beside
# the executable.
#
# This directory will eventually contain:
#
#   config.json
#   jarvis_conversation_memory.json
#
# The installer will create/configure config.json.
#
# ============================================================

APP_DATA_DIR = os.path.join(
    os.environ.get(
        "APPDATA",
        os.path.expanduser("~")
    ),
    "JARVIS"
)

os.makedirs(
    APP_DATA_DIR,
    exist_ok=True
)


CONFIG_FILE = os.path.join(
    APP_DATA_DIR,
    "config.json"
)

MEMORY_FILE = os.path.join(
    APP_DATA_DIR,
    "jarvis_conversation_memory.json"
)


# ============================================================
# AI PROVIDER CONFIGURATION
# ============================================================

SUPPORTED_PROVIDERS = (
    "gemini",
    "openai",
    "anthropic",
    "openrouter",
    "custom"
)


DEFAULT_AI_CONFIG = {

    "provider": "gemini",

    "api_key": "",

    "model": "gemini-2.5-flash",

    "base_url": "",

    "enabled": True
}


AI_CONFIG = {}


def load_ai_config():

    config = dict(
        DEFAULT_AI_CONFIG
    )

    # --------------------------------------------------------
    # Local configuration
    # --------------------------------------------------------

    try:

        if os.path.isfile(CONFIG_FILE):

            with open(
                CONFIG_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if isinstance(data, dict):

                ai = data.get(
                    "ai",
                    {}
                )

                if isinstance(ai, dict):

                    config.update(
                        ai
                    )

    except Exception as e:

        print(
            f"AI config load warning: {e}"
        )


    # --------------------------------------------------------
    # Environment-variable fallback
    #
    # Useful for development.
    # The public installer will use config.json.
    # --------------------------------------------------------

    provider_env = os.getenv(
        "JARVIS_AI_PROVIDER"
    )

    if provider_env:

        config["provider"] = (
            provider_env.strip().lower()
        )


    api_key_env = os.getenv(
        "JARVIS_AI_API_KEY"
    )

    if api_key_env:

        config["api_key"] = api_key_env


    # Backward compatibility with your
    # existing Gemini setup.
    if (
        not config.get("api_key")
        and config.get("provider") == "gemini"
    ):

        old_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if old_key:

            config["api_key"] = old_key


    model_env = os.getenv(
        "JARVIS_AI_MODEL"
    )

    if model_env:

        config["model"] = model_env


    base_url_env = os.getenv(
        "JARVIS_AI_BASE_URL"
    )

    if base_url_env:

        config["base_url"] = base_url_env


    provider = str(
        config.get(
            "provider",
            "gemini"
        )
    ).strip().lower()


    if provider not in SUPPORTED_PROVIDERS:

        provider = "gemini"


    config["provider"] = provider


    return config


AI_CONFIG = load_ai_config()


def save_ai_config(
    provider,
    api_key,
    model,
    base_url=""
):

    provider = str(
        provider
    ).strip().lower()


    if provider not in SUPPORTED_PROVIDERS:

        raise ValueError(
            f"Unsupported AI provider: {provider}"
        )


    config = {

        "provider": provider,

        "api_key": api_key.strip(),

        "model": model.strip(),

        "base_url": base_url.strip(),

        "enabled": True
    }


    root_config = {}

    if os.path.isfile(CONFIG_FILE):

        try:

            with open(
                CONFIG_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                root_config = json.load(f)

            if not isinstance(
                root_config,
                dict
            ):

                root_config = {}

        except Exception:

            root_config = {}


    root_config["ai"] = config


    temp_file = CONFIG_FILE + ".tmp"


    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            root_config,
            f,
            ensure_ascii=False,
            indent=2
        )


    os.replace(
        temp_file,
        CONFIG_FILE
    )


    global AI_CONFIG

    AI_CONFIG = config


def ai_provider():

    return str(
        AI_CONFIG.get(
            "provider",
            ""
        )
    ).lower().strip()


def ai_model():

    return str(
        AI_CONFIG.get(
            "model",
            ""
        )
    ).strip()


def ai_api_key():

    return str(
        AI_CONFIG.get(
            "api_key",
            ""
        )
    ).strip()


def ai_base_url():

    return str(
        AI_CONFIG.get(
            "base_url",
            ""
        )
    ).strip()


def ai_is_configured():

    return bool(
        ai_provider()
        and ai_model()
        and ai_api_key()
    )


# ============================================================
# SAFE PC POLICY
# ============================================================
#
# JARVIS IS INTENTIONALLY NON-DESTRUCTIVE.
#
# ALLOWED:
#   open
#   search
#   navigate
#   switch
#   close current window/tab
#   type
#   read/search filenames
#   media control
#   screenshots
#
# NOT IMPLEMENTED:
#   delete files
#   overwrite files
#   rename files
#   move files
#   uninstall software
#   format drives
#   registry modification
#   shutdown
#   restart
#   factory reset
#   credential extraction
#   disabling security
#   arbitrary shell commands
#
# ============================================================


# ============================================================
# CONVERSATION MEMORY
# ============================================================

CONVERSATION_HISTORY = []


def load_conversation_memory():

    try:

        if not os.path.exists(
            MEMORY_FILE
        ):

            return []


        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        if not isinstance(
            data,
            list
        ):

            return []


        history = []


        for item in data[-MAX_HISTORY:]:

            if (
                isinstance(item, dict)
                and isinstance(
                    item.get("question"),
                    str
                )
                and isinstance(
                    item.get("answer"),
                    str
                )
            ):

                history.append(
                    (
                        item["question"],
                        item["answer"]
                    )
                )


        print(
            f"MEMORY → LOADED "
            f"{len(history)} RECENT EXCHANGES"
        )


        return history


    except Exception as e:

        print(
            f"Memory load warning: {e}"
        )

        return []


def should_store_exchange(question):

    sensitive_words = (

        "password",
        "passcode",
        "api key",
        "apikey",
        "secret key",
        "credit card",
        "otp",
        "one time password",
        "private key"
    )


    text = question.lower().strip()


    return not any(
        word in text
        for word in sensitive_words
    )


def save_conversation_memory():

    try:

        data = []


        for question, answer in (
            CONVERSATION_HISTORY[-MAX_HISTORY:]
        ):

            data.append({

                "question": question,

                "answer": answer
            })


        temp_file = MEMORY_FILE + ".tmp"


        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )


        os.replace(
            temp_file,
            MEMORY_FILE
        )


    except Exception as e:

        print(
            f"Memory save warning: {e}"
        )


def remember_exchange(
    question,
    answer
):

    if not should_store_exchange(
        question
    ):

        print(
            "MEMORY → SENSITIVE COMMAND NOT STORED"
        )

        return


    CONVERSATION_HISTORY.append(
        (
            question,
            answer
        )
    )


    if len(
        CONVERSATION_HISTORY
    ) > MAX_HISTORY:

        CONVERSATION_HISTORY.pop(0)


    save_conversation_memory()


CONVERSATION_HISTORY = (
    load_conversation_memory()
)


# ============================================================
# AI REQUEST HELPERS
# ============================================================

def clean_ai_text(text):

    if text is None:

        return ""


    text = str(
        text
    ).strip()


    return text


def http_json_request(
    url,
    headers,
    payload,
    timeout=60
):

    data = json.dumps(
        payload
    ).encode(
        "utf-8"
    )


    request = urllib.request.Request(

        url,

        data=data,

        headers={
            **headers,
            "Content-Type":
                "application/json"
        },

        method="POST"
    )


    try:

        with urllib.request.urlopen(
            request,
            timeout=timeout
        ) as response:

            raw = response.read().decode(
                "utf-8"
            )

            return json.loads(
                raw
            )


    except urllib.error.HTTPError as e:

        try:

            error_body = e.read().decode(
                "utf-8",
                errors="replace"
            )

        except Exception:

            error_body = str(e)


        raise RuntimeError(
            f"HTTP {e.code}: {error_body}"
        )


    except urllib.error.URLError as e:

        raise RuntimeError(
            f"Connection error: {e}"
        )


def openai_compatible_response(
    base_url,
    api_key,
    model,
    messages
):

    base_url = base_url.rstrip(
        "/"
    )


    if not base_url.endswith(
        "/chat/completions"
    ):

        if base_url.endswith(
            "/v1"
        ):

            base_url += (
                "/chat/completions"
            )

        else:

            base_url += (
                "/v1/chat/completions"
            )


    payload = {

        "model": model,

        "messages": messages
    }


    data = http_json_request(

        base_url,

        {
            "Authorization":
                f"Bearer {api_key}"
        },

        payload
    )


    try:

        return clean_ai_text(
            data["choices"][0]["message"]["content"]
        )

    except Exception:

        raise RuntimeError(
            "The AI provider returned an unexpected response."
        )


# ============================================================
# GEMINI
# ============================================================

def ask_gemini(
    messages,
    model
):

    client = genai.Client(
        api_key=ai_api_key()
    )


    # Gemini's generate_content API accepts
    # a combined text prompt.
    #
    # We intentionally keep the existing
    # JARVIS behavior simple and portable.

    parts = []


    for message in messages:

        role = message.get(
            "role",
            "user"
        )

        content = message.get(
            "content",
            ""
        )


        if role == "system":

            parts.append(
                "SYSTEM:\n"
                + content
            )

        elif role == "assistant":

            parts.append(
                "JARVIS:\n"
                + content
            )

        else:

            parts.append(
                "USER:\n"
                + content
            )


    prompt = "\n\n".join(
        parts
    )


    response = client.models.generate_content(

        model=model,

        contents=prompt
    )


    return clean_ai_text(
        response.text
    )


# ============================================================
# OPENAI
# ============================================================

def ask_openai(
    messages,
    model
):

    return openai_compatible_response(

        "https://api.openai.com/v1",

        ai_api_key(),

        model,

        messages
    )


# ============================================================
# OPENROUTER
# ============================================================

def ask_openrouter(
    messages,
    model
):

    return openai_compatible_response(

        "https://openrouter.ai/api/v1",

        ai_api_key(),

        model,

        messages
    )


# ============================================================
# ANTHROPIC CLAUDE
# ============================================================

def ask_anthropic(
    messages,
    model
):

    system_messages = []

    normal_messages = []


    for message in messages:

        role = message.get(
            "role",
            "user"
        )

        content = message.get(
            "content",
            ""
        )


        if role == "system":

            system_messages.append(
                content
            )

        elif role in (
            "user",
            "assistant"
        ):

            normal_messages.append({

                "role": role,

                "content": content
            })


    payload = {

        "model": model,

        "max_tokens": 2048,

        "messages": normal_messages
    }


    if system_messages:

        payload["system"] = (
            "\n\n".join(
                system_messages
            )
        )


    data = http_json_request(

        "https://api.anthropic.com/v1/messages",

        {
            "x-api-key":
                ai_api_key(),

            "anthropic-version":
                "2023-06-01"
        },

        payload
    )


    try:

        content = data.get(
            "content",
            []
        )


        texts = []


        for item in content:

            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):

                texts.append(
                    item.get(
                        "text",
                        ""
                    )
                )


        return clean_ai_text(
            "\n".join(texts)
        )


    except Exception:

        raise RuntimeError(
            "Claude returned an unexpected response."
        )


# ============================================================
# CUSTOM / OPENAI-COMPATIBLE API
# ============================================================

def ask_custom(
    messages,
    model
):

    base_url = ai_base_url()


    if not base_url:

        raise RuntimeError(
            "Custom AI provider has no API endpoint configured."
        )


    return openai_compatible_response(

        base_url,

        ai_api_key(),

        model,

        messages
    )


# ============================================================
# UNIVERSAL AI ENGINE
# ============================================================

def ask_ai(
    messages
):

    provider = ai_provider()

    model = ai_model()

    key = ai_api_key()


    if not provider:

        return (
            "No AI provider has been configured yet."
        )


    if not key:

        return (
            "No AI API key has been configured yet."
        )


    if not model:

        return (
            "No AI model has been configured yet."
        )


    try:

        if provider == "gemini":

            return ask_gemini(
                messages,
                model
            )


        if provider == "openai":

            return ask_openai(
                messages,
                model
            )


        if provider == "anthropic":

            return ask_anthropic(
                messages,
                model
            )


        if provider == "openrouter":

            return ask_openrouter(
                messages,
                model
            )


        if provider == "custom":

            return ask_custom(
                messages,
                model
            )


        return (
            "The configured AI provider is not supported."
        )


    except Exception as e:

        print(
            f"AI error [{provider}]: {e}"
        )


        return (
            "I'm having trouble connecting "
            "to my AI system."
        )


# ============================================================
# CONTEXT RECALL
# ============================================================

CONTEXT_RECALL_PHRASES = (

    "remember what we were doing",
    "remember what we were working on",
    "what were we doing",
    "what were we working on",
    "what were we working on last",
    "what was i doing",
    "what was i working on",
    "where were we",
    "where did we leave off",
    "where did we stop",
    "what did we do last",
    "what did we talk about",
    "what were we talking about",
    "what was the last thing",
    "continue where we left off",
    "continue what we were doing",
    "continue from where we left off",
    "pick up where we left off",
    "do you remember what we were doing",
    "do you remember what we were working on"
)


def is_context_recall_question(
    question
):

    text = " ".join(
        question.lower().strip().split()
    )


    return any(
        phrase in text
        for phrase in CONTEXT_RECALL_PHRASES
    )


def format_history_for_ai(
    limit=20
):

    recent = (
        CONVERSATION_HISTORY[-limit:]
    )


    if not recent:

        return (
            "No previous JARVIS conversation is stored."
        )


    lines = []


    for index, (
        question,
        answer
    ) in enumerate(
        recent,
        1
    ):

        lines.append(

            f"{index}. User: {question}\n"
            f"   JARVIS: {answer}"
        )


    return "\n".join(
        lines
    )


def ask_context_recall(
    question
):

    if not CONVERSATION_HISTORY:

        return (
            "I don't have any previous JARVIS "
            "conversation stored yet."
        )


    history = format_history_for_ai(
        limit=20
    )


    messages = [

        {
            "role": "system",

            "content": (
                "You are JARVIS answering a "
                "context-recall question. "
                "Use ONLY the supplied JARVIS "
                "conversation history. "
                "Identify the user's most recent "
                "actual task or topic. "
                "Do not invent information. "
                "Answer naturally in 1-3 concise sentences.\n\n"
                "JARVIS CONVERSATION HISTORY:\n"
                f"{history}"
            )
        },

        {
            "role": "user",

            "content": question
        }
    ]


    answer = ask_ai(
        messages
    )


    if answer:

        return answer


    last_question, _ = (
        CONVERSATION_HISTORY[-1]
    )


    return (
        "The most recent thing I have in "
        "my JARVIS memory is that you asked: "
        f"'{last_question}'."
    )


# ============================================================
# WEB BRIDGE STATE
# ============================================================

edith_state = {

    "state": "ready",

    "message":
        "JARVIS CORE // ONLINE"
}


state_lock = threading.Lock()


def set_state(
    state,
    message
):

    with state_lock:

        edith_state["state"] = state

        edith_state["message"] = message


    print(
        f"WEB STATE → "
        f"{state.upper()} | {message}"
    )


# ============================================================
# WINDOWS KEYBOARD ENGINE
# ============================================================

KEYEVENTF_KEYUP = 0x0002


VK = {

    "backspace": 0x08,

    "tab": 0x09,

    "enter": 0x0D,

    "shift": 0x10,

    "ctrl": 0x11,

    "alt": 0x12,

    "escape": 0x1B,

    "space": 0x20,

    "left": 0x25,

    "up": 0x26,

    "right": 0x27,

    "down": 0x28,

    "home": 0x24,

    "end": 0x23,

    "pageup": 0x21,

    "pagedown": 0x22,

    "insert": 0x2D,

    "delete": 0x2E,

    "a": 0x41,

    "c": 0x43,

    "f": 0x46,

    "l": 0x4C,

    "r": 0x52,

    "t": 0x54,

    "v": 0x56,

    "w": 0x57,

    "x": 0x58,

    "z": 0x5A,

    "win": 0x5B,

    "volume_down": 0xAE,

    "volume_up": 0xAF,

    "media_play_pause": 0xB3,

    "snapshot": 0x2C
}


def key_down(
    vk_code
):

    ctypes.windll.user32.keybd_event(
        vk_code,
        0,
        0,
        0
    )


def key_up(
    vk_code
):

    ctypes.windll.user32.keybd_event(
        vk_code,
        0,
        KEYEVENTF_KEYUP,
        0
    )


def press_key(
    vk_code
):

    key_down(
        vk_code
    )

    time.sleep(
        0.03
    )

    key_up(
        vk_code
    )


def hotkey(
    *keys
):

    codes = [

        VK[key]
        if isinstance(
            key,
            str
        )
        else key

        for key in keys
    ]


    for code in codes:

        key_down(
            code
        )


    time.sleep(
        0.05
    )


    for code in reversed(
        codes
    ):

        key_up(
            code
        )


# ============================================================
# WINDOWS VOLUME
# ============================================================

def windows_volume_up():

    press_key(
        VK["volume_up"]
    )

    return True


def windows_volume_down():

    press_key(
        VK["volume_down"]
    )

    return True


# ============================================================
# SCREENSHOT
# ============================================================

def windows_screenshot():

    press_key(
        VK["snapshot"]
    )

    print(
        "SCREENSHOT → PRINT SCREEN"
    )

    return True


# ============================================================
# BROWSER NAVIGATION
# ============================================================

def browser_next_tab():

    hotkey(
        "ctrl",
        "tab"
    )

    print(
        "BROWSER → NEXT TAB"
    )

    return True


def browser_previous_tab():

    hotkey(
        "ctrl",
        "shift",
        "tab"
    )

    print(
        "BROWSER → PREVIOUS TAB"
    )

    return True


def browser_new_tab():

    hotkey(
        "ctrl",
        "t"
    )

    return True


def browser_close_tab():

    hotkey(
        "ctrl",
        "w"
    )

    return True


def browser_back():

    hotkey(
        "alt",
        "left"
    )

    return True


def browser_forward():

    hotkey(
        "alt",
        "right"
    )

    return True


def browser_refresh():

    hotkey(
        "ctrl",
        "r"
    )

    return True


def browser_find():

    hotkey(
        "ctrl",
        "f"
    )

    return True


# ============================================================
# WINDOW NAVIGATION
# ============================================================

def switch_window():

    hotkey(
        "alt",
        "tab"
    )

    return True


def close_current_window():

    hotkey(
        "alt",
        "f4"
    )

    return True


def maximize_window():

    hotkey(
        "win",
        "up"
    )

    return True


def minimize_window():

    hotkey(
        "win",
        "down"
    )

    return True


# ============================================================
# SAFE URL HANDLING
# ============================================================

def normalize_url(
    url
):

    url = url.strip()


    if not url:

        return None


    if not re.match(
        r"^https?://",
        url,
        re.IGNORECASE
    ):

        url = "https://" + url


    parsed = urlparse(
        url
    )


    if not parsed.netloc:

        return None


    return url


def open_url(
    url,
    browser=None
):

    url = normalize_url(
        url
    )


    if not url:

        return False


    if browser == "chrome":

        executable = (
            find_browser_executable(
                "chrome"
            )
        )


        if executable:

            subprocess.Popen(

                [
                    executable,
                    url
                ],

                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0
                )
            )

            return True


    if browser == "edge":

        executable = (
            find_browser_executable(
                "edge"
            )
        )


        if executable:

            subprocess.Popen(

                [
                    executable,
                    url
                ],

                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0
                )
            )

            return True


    return webbrowser.open(
        url
    )


# ============================================================
# BROWSER DISCOVERY
# ============================================================

def find_browser_executable(
    browser
):

    candidates = []


    if browser == "chrome":

        candidates = [

            os.path.expandvars(
                r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"
            ),

            os.path.expandvars(
                r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
            ),

            os.path.expandvars(
                r"%LocalAppData%\Google\Chrome\Application\chrome.exe"
            )
        ]


    elif browser == "edge":

        candidates = [

            os.path.expandvars(
                r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
            ),

            os.path.expandvars(
                r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
            ),

            os.path.expandvars(
                r"%LocalAppData%\Microsoft\Edge\Application\msedge.exe"
            )
        ]


    for path in candidates:

        if os.path.isfile(
            path
        ):

            return path


    return shutil.which(

        "chrome.exe"

        if browser == "chrome"

        else "msedge.exe"
    )


# ============================================================
# COMMON WEBSITES
# ============================================================

WEBSITES = {

    "youtube":
        "https://www.youtube.com",

    "google":
        "https://www.google.com",

    "gmail":
        "https://mail.google.com",

    "instagram":
        "https://www.instagram.com",

    "facebook":
        "https://www.facebook.com",

    "github":
        "https://github.com",

    "chatgpt":
        "https://chatgpt.com",

    "claude":
        "https://claude.ai"
}


def open_website(
    site,
    browser=None
):

    site = site.lower().strip()


    url = WEBSITES.get(
        site
    )


    if not url:

        return False


    return open_url(
        url,
        browser
    )


# ============================================================
# SEARCH
# ============================================================

def google_search(
    query,
    browser=None
):

    query = query.strip()


    if not query:

        return False


    url = (
        "https://www.google.com/search?q="
        + quote_plus(query)
    )


    return open_url(
        url,
        browser
    )


def youtube_search(
    query,
    browser=None
):

    query = query.strip()


    if not query:

        return False


    url = (
        "https://www.youtube.com/results?search_query="
        + quote_plus(query)
    )


    return open_url(
        url,
        browser
    )


# ============================================================
# SAFE APP LAUNCHING
# ============================================================

def launch_application(
    app
):

    app = app.lower().strip()


    if app == "chrome":

        executable = (
            find_browser_executable(
                "chrome"
            )
        )


        if executable:

            subprocess.Popen(
                [executable]
            )

            return True


        return False


    if app == "edge":

        executable = (
            find_browser_executable(
                "edge"
            )
        )


        if executable:

            subprocess.Popen(
                [executable]
            )

            return True


        return False


    if app == "notepad":

        subprocess.Popen(
            ["notepad.exe"]
        )

        return True


    if app == "calculator":

        subprocess.Popen(
            ["calc.exe"]
        )

        return True


    if app in (
        "file explorer",
        "explorer",
        "files"
    ):

        subprocess.Popen(
            ["explorer.exe"]
        )

        return True


    return False


# ============================================================
# SAFE FOLDER OPENING
# ============================================================

def open_folder(
    path
):

    path = os.path.expandvars(
        os.path.expanduser(
            path
        )
    )


    if not os.path.isdir(
        path
    ):

        return False


    subprocess.Popen(
        [
            "explorer.exe",
            path
        ]
    )


    return True


def common_folder(
    name
):

    home = os.path.expanduser(
        "~"
    )


    folders = {

        "desktop":
            os.path.join(
                home,
                "Desktop"
            ),

        "documents":
            os.path.join(
                home,
                "Documents"
            ),

        "downloads":
            os.path.join(
                home,
                "Downloads"
            ),

        "pictures":
            os.path.join(
                home,
                "Pictures"
            ),

        "videos":
            os.path.join(
                home,
                "Videos"
            ),

        "music":
            os.path.join(
                home,
                "Music"
            )
    }


    return folders.get(
        name.lower()
    )


# ============================================================
# SAFE FILE SEARCH
# ============================================================

def search_files(
    query,
    max_results=20
):

    query = query.lower().strip()


    if not query:

        return []


    home = os.path.expanduser(
        "~"
    )


    roots = [

        os.path.join(
            home,
            "Desktop"
        ),

        os.path.join(
            home,
            "Documents"
        ),

        os.path.join(
            home,
            "Downloads"
        ),

        os.path.join(
            home,
            "Pictures"
        ),

        os.path.join(
            home,
            "Videos"
        ),

        os.path.join(
            home,
            "Music"
        )
    ]


    results = []


    for root in roots:

        if not os.path.isdir(
            root
        ):

            continue


        for current_root, dirs, files in os.walk(
            root,
            topdown=True
        ):

            dirs[:] = [

                d for d in dirs

                if not d.startswith(".")

                and d.lower() not in (
                    "appdata",
                    "node_modules",
                    "__pycache__"
                )
            ]


            for filename in files:

                if query in filename.lower():

                    results.append(
                        os.path.join(
                            current_root,
                            filename
                        )
                    )


                    if len(
                        results
                    ) >= max_results:

                        return results


    return results


def open_file(
    path
):

    path = os.path.expandvars(
        os.path.expanduser(
            path
        )
    )


    if not os.path.isfile(
        path
    ):

        return False


    os.startfile(
        path
    )


    return True


# ============================================================
# FILE COMMAND EXTRACTION
# ============================================================

def extract_explicit_path(
    command
):

    quoted = re.findall(
        r'"([^"]+)"',
        command
    )


    for item in quoted:

        expanded = os.path.expandvars(
            os.path.expanduser(
                item
            )
        )


        if os.path.isfile(
            expanded
        ):

            return expanded


    match = re.search(

        r'([A-Za-z]:\\[^<>:"|?*\r\n]+)',

        command
    )


    if match:

        path = os.path.expandvars(
            os.path.expanduser(
                match.group(1).strip()
            )
        )


        if os.path.isfile(
            path
        ):

            return path


    return None


# ============================================================
# MEDIA SESSION
# ============================================================

def windows_media_session_command(
    action
):

    action = action.lower().strip()


    if action not in (
        "pause",
        "play"
    ):

        return False


    powershell_script = r'''
$ErrorActionPreference = "Stop"

try {

    Add-Type -AssemblyName System.Runtime.WindowsRuntime

    $managerType =
        [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager,
        Windows,
        ContentType=WindowsRuntime]

    $request = $managerType::RequestAsync()

    $manager =
        [System.WindowsRuntimeSystemExtensions]::AsTask(
            $request
        ).GetAwaiter().GetResult()

    if ($null -eq $manager) {
        exit 2
    }

    $session = $manager.GetCurrentSession()

    if ($null -eq $session) {
        exit 2
    }

    $operation = $null

    if ("ACTION" -eq "pause") {

        $operation = $session.TryPauseAsync()

    }
    elseif ("ACTION" -eq "play") {

        $operation = $session.TryPlayAsync()

    }

    if ($null -eq $operation) {
        exit 4
    }

    $result =
        [System.WindowsRuntimeSystemExtensions]::AsTask(
            $operation
        ).GetAwaiter().GetResult()

    if ($result) {
        exit 0
    }

    exit 3

}
catch {

    exit 5
}
'''


    powershell_script = (
        powershell_script.replace(
            "ACTION",
            action
        )
    )


    try:

        result = subprocess.run(

            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                powershell_script
            ],

            capture_output=True,

            text=True,

            timeout=5,

            creationflags=getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0
            )
        )


        return (
            result.returncode == 0
        )


    except Exception as e:

        print(
            f"Media session error: {e}"
        )

        return False


# ============================================================
# MEDIA COMMANDS
# ============================================================

def handle_media_command(
    action
):

    action = action.lower().strip()


    if action == "pause":

        return windows_media_session_command(
            "pause"
        )


    if action == "play":

        return windows_media_session_command(
            "play"
        )


    if action == "volume_up":

        return windows_volume_up()


    if action == "volume_down":

        return windows_volume_down()


    if action == "screenshot":

        return windows_screenshot()


    if action == "next_tab":

        return browser_next_tab()


    if action == "previous_tab":

        return browser_previous_tab()


    return False


# ============================================================
# TEXT TYPING
# ============================================================

def type_text(
    text
):

    if not text:

        return False


    user32 = ctypes.windll.user32

    KEYEVENTF_UNICODE = 0x0004


    for char in text:

        code = ord(char)


        user32.keybd_event(
            0,
            code,
            KEYEVENTF_UNICODE,
            0
        )


        user32.keybd_event(
            0,
            code,
            KEYEVENTF_UNICODE |
            KEYEVENTF_KEYUP,
            0
        )


        time.sleep(
            0.003
        )


    return True


# ============================================================
# SAFE COMMAND ROUTER
# ============================================================

def pc_action(
    command
):

    cmd = command.lower().strip()


    destructive_patterns = (

        r"\bdelete\b",
        r"\berase\b",
        r"\bremove file\b",
        r"\bformat\b",
        r"\buninstall\b",
        r"\bshutdown\b",
        r"\brestart\b",
        r"\breboot\b",
        r"\bfactory reset\b",
        r"\breset windows\b",
        r"\bkill all\b",
        r"\bterminate all\b",
        r"\bdisable antivirus\b",
        r"\bdisable defender\b",
        r"\bchange registry\b",
        r"\bedit registry\b"
    )


    for pattern in destructive_patterns:

        if re.search(
            pattern,
            cmd
        ):

            speak(
                "I won't perform destructive system actions."
            )

            return True


    # --------------------------------------------------------
    # MEDIA
    # --------------------------------------------------------

    if any(
        phrase in cmd
        for phrase in (
            "pause",
            "pause the music",
            "pause video"
        )
    ):

        if handle_media_command(
            "pause"
        ):

            speak(
                "Paused."
            )

            return True


    if any(
        phrase in cmd
        for phrase in (
            "play",
            "resume",
            "resume the music",
            "resume video"
        )
    ):

        if handle_media_command(
            "play"
        ):

            speak(
                "Playing."
            )

            return True


    if (
        "volume up" in cmd
        or "increase volume" in cmd
        or "turn volume up" in cmd
    ):

        handle_media_command(
            "volume_up"
        )

        speak(
            "Volume increased."
        )

        return True


    if (
        "volume down" in cmd
        or "decrease volume" in cmd
        or "turn volume down" in cmd
    ):

        handle_media_command(
            "volume_down"
        )

        speak(
            "Volume decreased."
        )

        return True


    if (
        "take a screenshot" in cmd
        or "take screenshot" in cmd
        or "screenshot" == cmd
    ):

        handle_media_command(
            "screenshot"
        )

        speak(
            "Screenshot taken."
        )

        return True


    # --------------------------------------------------------
    # BROWSER
    # --------------------------------------------------------

    if (
        "next tab" in cmd
        or "next browser tab" in cmd
    ):

        browser_next_tab()

        speak(
            "Next tab."
        )

        return True


    if (
        "previous tab" in cmd
        or "previous browser tab" in cmd
    ):

        browser_previous_tab()

        speak(
            "Previous tab."
        )

        return True


    if (
        "new tab" in cmd
        or "open a new tab" in cmd
    ):

        browser_new_tab()

        speak(
            "New tab."
        )

        return True


    if (
        "close this tab" in cmd
        or "close the tab" in cmd
    ):

        browser_close_tab()

        speak(
            "Tab closed."
        )

        return True


    if cmd in (
        "go back",
        "back",
        "browser back"
    ):

        browser_back()

        speak(
            "Going back."
        )

        return True


    if cmd in (
        "go forward",
        "forward",
        "browser forward"
    ):

        browser_forward()

        speak(
            "Going forward."
        )

        return True


    if (
        "refresh page" in cmd
        or cmd == "refresh"
    ):

        browser_refresh()

        speak(
            "Refreshing."
        )

        return True


    # --------------------------------------------------------
    # WINDOW
    # --------------------------------------------------------

    if cmd in (
        "switch window",
        "switch app",
        "switch applications"
    ):

        switch_window()

        speak(
            "Switching."
        )

        return True


    if (
        "close this window" in cmd
        or cmd == "close window"
    ):

        close_current_window()

        speak(
            "Window closed."
        )

        return True


    if (
        "maximize window" in cmd
        or cmd == "maximize"
    ):

        maximize_window()

        speak(
            "Maximized."
        )

        return True


    if (
        "minimize window" in cmd
        or cmd == "minimize"
    ):

        minimize_window()

        speak(
            "Minimized."
        )

        return True


    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    keyboard_actions = {

        "press enter":
            ("enter", "Enter pressed."),

        "press escape":
            ("escape", "Escape pressed."),

        "press tab":
            ("tab", "Tab pressed."),

        "copy":
            ("copy", "Copied."),

        "paste":
            ("paste", "Pasted."),

        "cut":
            ("cut", "Cut."),

        "select all":
            ("selectall", "Selected all."),

        "undo":
            ("undo", "Undone."),

        "find":
            ("find", "Find opened.")
    }


    if cmd in keyboard_actions:

        action, response = (
            keyboard_actions[cmd]
        )


        if action == "enter":

            press_key(
                VK["enter"]
            )


        elif action == "escape":

            press_key(
                VK["escape"]
            )


        elif action == "tab":

            press_key(
                VK["tab"]
            )


        elif action == "copy":

            hotkey(
                "ctrl",
                "c"
            )


        elif action == "paste":

            hotkey(
                "ctrl",
                "v"
            )


        elif action == "cut":

            hotkey(
                "ctrl",
                "x"
            )


        elif action == "selectall":

            hotkey(
                "ctrl",
                "a"
            )


        elif action == "undo":

            hotkey(
                "ctrl",
                "z"
            )


        elif action == "find":

            browser_find()


        speak(
            response
        )

        return True


    # --------------------------------------------------------
    # TYPE TEXT
    # --------------------------------------------------------

    type_match = re.match(

        r'^(?:type|write)\s+["\'](.+)["\']$',

        command,

        re.IGNORECASE
    )


    if type_match:

        text = type_match.group(
            1
        )


        type_text(
            text
        )


        speak(
            "Done."
        )


        return True


    # --------------------------------------------------------
    # EXACT FILE
    # --------------------------------------------------------

    explicit_path = (
        extract_explicit_path(
            command
        )
    )


    if explicit_path:

        if open_file(
            explicit_path
        ):

            speak(
                "Opening the file."
            )

        else:

            speak(
                "I couldn't open that file."
            )


        return True


    # --------------------------------------------------------
    # FOLDERS
    # --------------------------------------------------------

    for folder_name in (

        "desktop",
        "documents",
        "downloads",
        "pictures",
        "videos",
        "music"
    ):

        if (
            folder_name in cmd
            and (
                "open" in cmd
                or "show" in cmd
                or "go to" in cmd
            )
        ):

            path = common_folder(
                folder_name
            )


            if (
                path
                and open_folder(path)
            ):

                speak(
                    f"Opening {folder_name}."
                )

                return True


    # --------------------------------------------------------
    # FILE SEARCH
    # --------------------------------------------------------

    search_file_match = re.search(

        r"(?:find|search for|look for|locate)\s+"
        r"(?:my\s+)?(.+?)"
        r"(?:\s+file|\s+document|\s+pdf)?$",

        command,

        re.IGNORECASE
    )


    if search_file_match:

        query = (
            search_file_match
            .group(1)
            .strip()
        )


        results = search_files(
            query
        )


        if not results:

            speak(
                f"I couldn't find a file "
                f"matching {query}."
            )

            return True


        if len(results) == 1:

            speak(
                "I found it. Opening "
                f"{os.path.basename(results[0])}."
            )


            open_file(
                results[0]
            )


            return True


        best = results[0]


        speak(

            f"I found {len(results)} matches. "
            f"Opening {os.path.basename(best)}."
        )


        open_file(
            best
        )


        return True


    # --------------------------------------------------------
    # GOOGLE SEARCH
    # --------------------------------------------------------

    google_match = re.search(

        r"(?:search google for|google search for|search google)\s+(.+)",

        command,

        re.IGNORECASE
    )


    if google_match:

        query = (
            google_match
            .group(1)
            .strip()
        )


        google_search(
            query
        )


        speak(
            f"Searching Google for {query}."
        )


        return True


    # --------------------------------------------------------
    # YOUTUBE SEARCH
    # --------------------------------------------------------

    youtube_match = re.search(

        r"(?:search youtube for|search youtube|find on youtube)\s+(.+)",

        command,

        re.IGNORECASE
    )


    if youtube_match:

        query = (
            youtube_match
            .group(1)
            .strip()
        )


        youtube_search(
            query
        )


        speak(
            f"Searching YouTube for {query}."
        )


        return True


    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    url_match = re.search(

        r"(https?://[^\s]+|www\.[^\s]+)",

        command,

        re.IGNORECASE
    )


    if url_match:

        url = url_match.group(
            1
        )


        browser = None


        if "edge" in cmd:

            browser = "edge"


        elif "chrome" in cmd:

            browser = "chrome"


        open_url(
            url,
            browser
        )


        speak(
            "Opening the website."
        )


        return True


    # --------------------------------------------------------
    # WEBSITES
    # --------------------------------------------------------

    for site in WEBSITES:

        if site in cmd:

            browser = None


            if "edge" in cmd:

                browser = "edge"


            elif "chrome" in cmd:

                browser = "chrome"


            if (
                "open" in cmd
                or "launch" in cmd
                or "start" in cmd
                or "go to" in cmd
            ):

                if open_website(
                    site,
                    browser
                ):

                    speak(
                        f"Opening {site}."
                    )


                    return True


    # --------------------------------------------------------
    # APPLICATIONS
    # --------------------------------------------------------

    app_aliases = {

        "chrome":
            "chrome",

        "google chrome":
            "chrome",

        "edge":
            "edge",

        "microsoft edge":
            "edge",

        "notepad":
            "notepad",

        "calculator":
            "calculator",

        "calc":
            "calculator",

        "file explorer":
            "file explorer",

        "explorer":
            "file explorer",

        "files":
            "file explorer"
    }


    for phrase, app in (
        app_aliases.items()
    ):

        if (
            phrase in cmd
            and (
                "open" in cmd
                or "launch" in cmd
                or "start" in cmd
            )
        ):

            if launch_application(
                app
            ):

                speak(
                    f"Opening {phrase}."
                )

            else:

                speak(
                    f"I couldn't find "
                    f"{phrase} on this PC."
                )


            return True


    # --------------------------------------------------------
    # SCROLL
    # --------------------------------------------------------

    if (
        "scroll down" in cmd
        or "scroll lower" in cmd
    ):

        press_key(
            VK["pagedown"]
        )


        speak(
            "Scrolling down."
        )


        return True


    if (
        "scroll up" in cmd
        or "scroll higher" in cmd
    ):

        press_key(
            VK["pageup"]
        )


        speak(
            "Scrolling up."
        )


        return True


    return False


# ============================================================
# WEB BRIDGE
# ============================================================

class EdithBridgeHandler(
    BaseHTTPRequestHandler
):

    # --------------------------------------------------------
    # CORS / PREFLIGHT
    # --------------------------------------------------------

    def do_OPTIONS(self):

        self.send_response(
            204
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Private-Network",
            "true"
        )

        self.end_headers()


    # --------------------------------------------------------
    # JSON RESPONSE
    # --------------------------------------------------------

    def send_json(
        self,
        data,
        status=200
    ):

        response = json.dumps(
            data
        ).encode(
            "utf-8"
        )

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Private-Network",
            "true"
        )

        self.send_header(
            "Cache-Control",
            "no-store"
        )

        self.end_headers()

        self.wfile.write(
            response
        )


    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path


        # ----------------------------------------------------
        # AI CONFIGURATION
        # ----------------------------------------------------

        if path == "/config":

            try:

                content_length = int(
                    self.headers.get(
                        "Content-Length",
                        "0"
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                content_length = 0


            if content_length <= 0:

                self.send_json(
                    {
                        "ok": False,
                        "error": "Empty request body."
                    },
                    400
                )

                return


            if content_length > 32 * 1024:

                self.send_json(
                    {
                        "ok": False,
                        "error": "Request too large."
                    },
                    413
                )

                return


            try:

                raw_body = self.rfile.read(
                    content_length
                )

                data = json.loads(
                    raw_body.decode(
                        "utf-8"
                    )
                )

            except json.JSONDecodeError:

                self.send_json(
                    {
                        "ok": False,
                        "error": "Invalid JSON."
                    },
                    400
                )

                return

            except Exception:

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "Could not read request."
                    },
                    400
                )

                return


            if not isinstance(
                data,
                dict
            ):

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "Invalid configuration."
                    },
                    400
                )

                return


            provider = data.get(
                "provider",
                ""
            )

            api_key = data.get(
                "api_key",
                ""
            )

            model = data.get(
                "model",
                ""
            )

            base_url = data.get(
                "base_url",
                ""
            )


            if not isinstance(
                provider,
                str
            ):

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "Provider must be text."
                    },
                    400
                )

                return


            if not isinstance(
                api_key,
                str
            ):

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "API key must be text."
                    },
                    400
                )

                return


            if not isinstance(
                model,
                str
            ):

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "AI model must be text."
                    },
                    400
                )

                return


            if not isinstance(
                base_url,
                str
            ):

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "Base URL must be text."
                    },
                    400
                )

                return


            provider = provider.strip().lower()
            api_key = api_key.strip()
            model = model.strip()
            base_url = base_url.strip()


            if provider not in SUPPORTED_PROVIDERS:

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "Unsupported AI provider."
                    },
                    400
                )

                return


            if not api_key:

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "API key is required."
                    },
                    400
                )

                return


            if not model:

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "AI model is required."
                    },
                    400
                )

                return


            if (
                provider == "custom"
                and not base_url
            ):

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "Custom provider requires "
                            "an API endpoint."
                    },
                    400
                )

                return


            try:

                save_ai_config(
                    provider,
                    api_key,
                    model,
                    base_url
                )

            except Exception as e:

                print(
                    f"AI config save error: {e}"
                )

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "Could not save "
                            "AI configuration."
                    },
                    500
                )

                return


            self.send_json({

                "ok":
                    True,

                "provider":
                    ai_provider(),

                "model":
                    ai_model(),

                "configured":
                    ai_is_configured()

            })


            print(

                "AI CONFIG → "
                f"{ai_provider()} / "
                f"{ai_model()} configured"
            )


            return


        # ----------------------------------------------------
        # UNKNOWN POST
        # ----------------------------------------------------

        self.send_json(

            {
                "ok": False,
                "error": "Not found"
            },

            404
        )


    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path


        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        if path == "/state":

            with state_lock:

                data = dict(
                    edith_state
                )


            self.send_json(
                data
            )

            return


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if path == "/status":

            self.send_json({

                "installed":
                    True,

                "running":
                    True,

                "version":
                    APP_VERSION,

                "state":
                    edith_state["state"],

                "message":
                    edith_state["message"],

                "pc_control":
                    True,

                "destructive_actions":
                    False,

                "ai_provider":
                    ai_provider(),

                "ai_model":
                    ai_model(),

                "ai_configured":
                    ai_is_configured()

            })

            return


        # ----------------------------------------------------
        # MEDIA
        # ----------------------------------------------------

        if path == "/media":

            query = parse_qs(
                parsed.query
            )

            action = query.get(
                "action",
                [""]
            )[0].lower().strip()


            allowed = (

                "pause",

                "play",

                "volume_up",

                "volume_down",

                "screenshot",

                "next_tab",

                "previous_tab"

            )


            if action not in allowed:

                self.send_json(

                    {
                        "ok": False,
                        "error":
                            "Invalid action"
                    },

                    400
                )

                return


            success = (
                handle_media_command(
                    action
                )
            )


            self.send_json(

                {
                    "ok":
                        success,

                    "action":
                        action

                },

                200 if success else 500
            )

            return


        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        self.send_json(

            {
                "ok": False,
                "error": "Not found"
            },

            404
        )


    # --------------------------------------------------------
    # SILENT SERVER LOG
    # --------------------------------------------------------

    def log_message(
        self,
        format,
        *args
    ):

        return


# ============================================================
# START BRIDGE
# ============================================================

def start_bridge():

    try:

        server = HTTPServer(

            (
                BRIDGE_HOST,
                BRIDGE_PORT
            ),

            EdithBridgeHandler
        )


        print(

            f"JARVIS WEB BRIDGE: "
            f"http://{BRIDGE_HOST}:{BRIDGE_PORT}"
        )


        server.serve_forever()


    except Exception as e:

        print(
            f"Web bridge error: {e}"
        )


bridge_thread = threading.Thread(

    target=start_bridge,

    daemon=True
)


bridge_thread.start()


# ============================================================
# AUDIO
# ============================================================

pygame.mixer.init()

recognizer = sr.Recognizer()


# ============================================================
# DYNAMIC MICROPHONE
# ============================================================

def find_microphone():

    devices = sd.query_devices()


    preferred = [

        "microphone",

        "bassheads",

        "headset",

        "usb",

        "headphones"
    ]


    candidates = []


    for index, device in enumerate(
        devices
    ):

        if device[
            "max_input_channels"
        ] <= 0:

            continue


        name = device[
            "name"
        ].lower()


        score = 0


        for word in preferred:

            if word in name:

                score += 1


        candidates.append(

            (
                score,
                index,
                device
            )
        )


    if not candidates:

        raise RuntimeError(
            "No microphone found."
        )


    candidates.sort(

        key=lambda x: x[0],

        reverse=True
    )


    _, index, device = (
        candidates[0]
    )


    return index, device


# ============================================================
# TEXT TO SPEECH
# ============================================================

async def generate_voice(
    text,
    filename
):

    communicator = edge_tts.Communicate(

        text,

        VOICE
    )


    await communicator.save(
        filename
    )


def speak(
    text
):

    print(
        f"\nJARVIS: {text}\n"
    )


    filename = (
        tempfile.NamedTemporaryFile(

            suffix=".mp3",

            delete=False
        ).name
    )


    try:

        asyncio.run(

            generate_voice(

                text,

                filename
            )
        )


        pygame.mixer.music.load(
            filename
        )


        pygame.mixer.music.play()


        while pygame.mixer.music.get_busy():

            time.sleep(
                0.02
            )


        pygame.mixer.music.unload()


    except Exception as e:

        print(
            f"TTS error: {e}"
        )


    finally:

        try:

            os.remove(
                filename
            )

        except Exception:

            pass


# ============================================================
# YES VOICE
# ============================================================

YES_FILE = os.path.join(

    tempfile.gettempdir(),

    "jarvis_yes.mp3"
)


async def create_yes_voice():

    communicator = edge_tts.Communicate(

        "Yes?",

        VOICE
    )


    await communicator.save(
        YES_FILE
    )


def prepare_yes_voice():

    if not os.path.exists(
        YES_FILE
    ):

        print(
            "Preparing Andrew voice..."
        )


        asyncio.run(
            create_yes_voice()
        )


def say_yes():

    print(
        "\nJARVIS: Yes?\n"
    )


    try:

        pygame.mixer.music.load(
            YES_FILE
        )


        pygame.mixer.music.play()


        while pygame.mixer.music.get_busy():

            time.sleep(
                0.02
            )


        pygame.mixer.music.unload()


    except Exception:

        speak(
            "Yes?"
        )


# ============================================================
# COMMAND LISTENER
# ============================================================

def listen_for_command():

    mic_index, mic_info = (
        find_microphone()
    )


    sample_rate = int(
        mic_info[
            "default_samplerate"
        ]
    )


    channels = (

        2

        if mic_info[
            "max_input_channels"
        ] >= 2

        else 1
    )


    print(

        f"Listening for your command "
        f"(up to {MAX_COMMAND_TIME:.0f} seconds)..."
    )


    try:

        calibration = sd.rec(

            int(
                sample_rate * 0.30
            ),

            samplerate=sample_rate,

            channels=channels,

            dtype="float32",

            device=mic_index
        )


        sd.wait()


        if channels > 1:

            calibration = (
                calibration.mean(
                    axis=1
                )
            )

        else:

            calibration = (
                calibration[:, 0]
            )


        noise_rms = float(

            np.sqrt(

                np.mean(
                    calibration ** 2
                )
            )
        )


    except Exception as e:

        print(
            f"Microphone error: {e}"
        )

        return ""


    threshold = max(

        0.012,

        noise_rms * 2.5
    )


    chunk_time = 0.05


    chunk_size = max(

        1,

        int(
            sample_rate *
            chunk_time
        )
    )


    maximum_chunks = int(

        MAX_COMMAND_TIME /
        chunk_time
    )


    silence_chunks_needed = max(

        1,

        int(
            END_SILENCE /
            chunk_time
        )
    )


    minimum_speech_chunks = max(

        1,

        int(
            MIN_SPEECH_TIME /
            chunk_time
        )
    )


    recorded = []

    speech_started = False

    speech_chunks = 0

    silent_chunks = 0


    try:

        with sd.InputStream(

            device=mic_index,

            samplerate=sample_rate,

            channels=channels,

            dtype="float32",

            blocksize=chunk_size

        ) as stream:

            for _ in range(
                maximum_chunks
            ):

                data, _ = stream.read(
                    chunk_size
                )


                data = np.asarray(
                    data
                )


                if channels > 1:

                    mono = data.mean(
                        axis=1
                    )

                else:

                    mono = data[:, 0]


                volume = float(

                    np.sqrt(

                        np.mean(
                            mono ** 2
                        )
                    )
                )


                if not speech_started:

                    if volume >= threshold:

                        speech_started = True

                        speech_chunks = 1

                        silent_chunks = 0

                        recorded.append(
                            mono.copy()
                        )

                    continue


                recorded.append(
                    mono.copy()
                )


                speech_chunks += 1


                if volume >= threshold:

                    silent_chunks = 0

                else:

                    silent_chunks += 1


                if (

                    speech_chunks >=
                    minimum_speech_chunks

                    and

                    silent_chunks >=
                    silence_chunks_needed

                ):

                    break


    except Exception as e:

        print(
            f"Microphone error: {e}"
        )

        return ""


    if not speech_started:

        print(
            "No command detected."
        )

        return ""


    audio = np.concatenate(
        recorded
    )


    audio = np.clip(
        audio,
        -1,
        1
    )


    wav_file = tempfile.NamedTemporaryFile(

        suffix=".wav",

        delete=False
    )


    wav_path = wav_file.name


    wav_file.close()


    try:

        sf.write(

            wav_path,

            audio,

            sample_rate,

            subtype="PCM_16"
        )


        with sr.AudioFile(
            wav_path
        ) as source:

            recorded_audio = (
                recognizer.record(
                    source
                )
            )


        try:

            text = (
                recognizer.recognize_google(

                    recorded_audio,

                    language="en-US"
                )
            )


            print(
                f"You: {text}"
            )


            return text.strip()


        except sr.UnknownValueError:

            print(
                "I couldn't understand that."
            )

            return ""


        except sr.RequestError as e:

            print(
                f"Speech recognition error: {e}"
            )

            return ""


    finally:

        try:

            os.remove(
                wav_path
            )

        except Exception:

            pass


# ============================================================
# GENERAL JARVIS RESPONSE
# ============================================================

def ask_edith(
    question
):

    history = format_history_for_ai(
        limit=12
    )


    messages = [

        {
            "role": "system",

            "content": (

                "You are JARVIS, a fast personal "
                "AI assistant. "
                "Answer naturally and directly. "
                "Keep answers concise unless asked "
                "for detail. "
                "Use recent JARVIS conversation "
                "as context. "
                "Do not claim to have performed "
                "a PC action unless the PC control "
                "layer actually performed it.\n\n"

                "RECENT JARVIS CONVERSATION:\n"
                f"{history}"
            )
        },

        {
            "role": "user",

            "content": question
        }
    ]


    return ask_ai(
        messages
    )


# ============================================================
# HEY JARVIS DETECTOR
# ============================================================

def wait_for_jarvis():

    mic_index, mic_info = (
        find_microphone()
    )


    mic_rate = int(
        mic_info[
            "default_samplerate"
        ]
    )


    channels = (

        2

        if mic_info[
            "max_input_channels"
        ] >= 2

        else 1
    )


    wake_model = Model(

        wakeword_models=[
            WAKE_WORD
        ],

        inference_framework="onnx"
    )


    detected = False


    def callback(
        indata,
        frames,
        callback_time,
        status
    ):

        nonlocal detected


        if detected:

            return


        audio = indata


        if audio.ndim > 1:

            audio = audio.mean(
                axis=1
            )


        if mic_rate != 16000:

            audio = resample_poly(

                audio,

                16000,

                mic_rate
            )


        audio = np.clip(

            audio,

            -1,

            1
        )


        audio = (

            audio * 32767

        ).astype(
            np.int16
        )


        prediction = wake_model.predict(
            audio
        )


        score = prediction.get(

            WAKE_WORD,

            0
        )


        if score >= WAKE_THRESHOLD:

            detected = True


    try:

        with sd.InputStream(

            device=mic_index,

            samplerate=mic_rate,

            channels=channels,

            dtype="float32",

            blocksize=2400,

            callback=callback

        ):

            while not detected:

                sd.sleep(
                    20
                )


    except Exception as e:

        print(
            f"\nWake detector error: {e}"
        )


        time.sleep(
            1
        )


        return False


    return True


# ============================================================
# STARTUP
# ============================================================

print()

print(
    "=========================================="
)

print(
    "          JARVIS SYSTEM ONLINE"
)

print(
    "=========================================="
)

print(
    f"Companion Version: {APP_VERSION}"
)

print(
    f"AI Provider: "
    f"{ai_provider().upper() or 'NOT CONFIGURED'}"
)

print(
    f"AI Model: "
    f"{ai_model() or 'NOT CONFIGURED'}"
)

print(
    "Andrew voice: ACTIVE"
)

print(
    "Dynamic microphone: ACTIVE"
)

print(
    "Wake word: HEY JARVIS"
)

print(
    "COMMAND WINDOW: 7 SECONDS MAX"
)

print(
    "CONTINUOUS CONVERSATION: ACTIVE"
)

print(
    "ADVANCED PC CONTROL: ACTIVE"
)

print(
    "SAFE / NON-DESTRUCTIVE MODE: ACTIVE"
)

print(
    "BROWSER CONTROL: ACTIVE"
)

print(
    "FILE SEARCH / OPEN: ACTIVE"
)

print(
    "WINDOW NAVIGATION: ACTIVE"
)

print(
    "KEYBOARD CONTROL: ACTIVE"
)

print(
    "EXTERNAL MEDIA CONTROL: ACTIVE"
)

print(
    "WEB CORE BRIDGE: ACTIVE"
)

print(
    "LOCAL INSTANT REPLIES: ACTIVE"
)

print(
    "PERSISTENT CONVERSATION MEMORY: ACTIVE"
)

print(
    "DESTRUCTIVE ACTIONS: BLOCKED"
)

print(
    "=========================================="
)

print()


if not ai_is_configured():

    print(
        "WARNING: AI PROVIDER IS NOT CONFIGURED."
    )

    print(
        "Configure the AI provider from "
        "the J.A.R.V.I.S. web interface."
    )

    print(
        "The Companion is waiting for "
        "AI configuration."
    )

    print()

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    try:

        set_state(
            "ready",
            "JARVIS CORE // ONLINE"
        )


        woke_up = wait_for_jarvis()


        if not woke_up:

            continue


        print(
            "\nHEY JARVIS detected!"
        )


        set_state(
            "listening",
            "VOICE INPUT // ACTIVE"
        )


        say_yes()


        while True:

            set_state(
                "listening",
                "VOICE INPUT // ACTIVE"
            )


            print(
                "\nListening..."
            )


            command = (
                listen_for_command()
            )


            if not command:

                set_state(
                    "ready",
                    "JARVIS CORE // ONLINE"
                )


                time.sleep(
                    COOLDOWN
                )


                print(
                    "\nJARVIS is sleeping."
                )


                print(
                    "Say: HEY JARVIS"
                )


                print()


                break


            set_state(
                "thinking",
                "NEURAL PROCESSING // ACTIVE"
            )


            # ------------------------------------------------
            # CONTEXT RECALL
            # ------------------------------------------------

            if is_context_recall_question(
                command
            ):

                answer = (
                    ask_context_recall(
                        command
                    )
                )


                set_state(
                    "responding",
                    "JARVIS RESPONSE // ACTIVE"
                )


                speak(
                    answer
                )


                remember_exchange(
                    command,
                    answer
                )


            else:

                # --------------------------------------------
                # LOCAL RESPONSE
                # --------------------------------------------

                local_answer = (
                    local_responses.get_response(

                        command,

                        CONVERSATION_HISTORY
                    )
                )


                if local_answer is not None:

                    set_state(
                        "responding",
                        "JARVIS RESPONSE // ACTIVE"
                    )


                    speak(
                        local_answer
                    )


                    remember_exchange(
                        command,
                        local_answer
                    )


                # --------------------------------------------
                # PC CONTROL
                # --------------------------------------------

                elif pc_action(
                    command
                ):

                    set_state(
                        "responding",
                        "PC ACTION // COMPLETE"
                    )


                    remember_exchange(

                        command,

                        "(performed a safe PC action)"
                    )


                # --------------------------------------------
                # AI
                # --------------------------------------------

                else:

                    print(
                        "\nJARVIS is thinking..."
                    )


                    answer = ask_edith(
                        command
                    )


                    set_state(
                        "responding",
                        "JARVIS RESPONSE // ACTIVE"
                    )


                    speak(
                        answer
                    )


                    remember_exchange(
                        command,
                        answer
                    )


            set_state(
                "ready",
                "JARVIS CORE // ONLINE"
            )


            time.sleep(
                COOLDOWN
            )


            print(
                "\nJARVIS ready for your next command..."
            )


    except KeyboardInterrupt:

        set_state(
            "offline",
            "JARVIS CORE // OFFLINE"
        )


        print(
            "\n\nJARVIS shutting down..."
        )


        break


    except Exception as e:

        print(
            f"\nUnexpected error: {e}"
        )


        set_state(
            "ready",
            "JARVIS CORE // ONLINE"
        )


        time.sleep(
            1
        )
