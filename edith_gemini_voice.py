import os
import tempfile
import time
import webbrowser
import subprocess
import asyncio
import threading
import json
import ctypes
from urllib.parse import urlparse, parse_qs
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
# JARVIS SETTINGS
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = "gemini-3.5-flash-lite"
VOICE = "en-US-AndrewNeural"

WAKE_WORD = "hey_jarvis"
WAKE_THRESHOLD = 0.50

MAX_COMMAND_TIME = 7.0
END_SILENCE = 1.0
MIN_SPEECH_TIME = 0.25
COOLDOWN = 1.5


# ============================================================
# CONVERSATION MEMORY
# ============================================================

# Persistent rolling memory. This is JARVIS' OWN conversation
# history — it does not read your ChatGPT history or any other
# assistant's chats. It lets JARVIS remember what YOU told JARVIS
# across restarts and answer context questions from its own history.

CONVERSATION_HISTORY = []
MAX_HISTORY = 40

MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "jarvis_conversation_memory.json"
)


def load_conversation_memory():

    try:

        if not os.path.exists(MEMORY_FILE):
            return []

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if not isinstance(data, list):
            return []

        history = []

        for item in data[-MAX_HISTORY:]:

            if (
                isinstance(item, dict)
                and isinstance(item.get("question"), str)
                and isinstance(item.get("answer"), str)
            ):

                history.append((
                    item["question"],
                    item["answer"]
                ))

        print(
