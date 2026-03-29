"""
Clipboard save/restore utility for dictation paste.

Saves the current clipboard contents before pasting transcribed text,
then restores the original contents afterward. This prevents dictation
from overwriting whatever the user had previously copied.
"""

import time

import pyperclip
from pynput import keyboard as pykbd


def paste_with_restore(text: str) -> None:
    """Copy text to clipboard, simulate Ctrl+V, then restore original clipboard."""
    original = pyperclip.paste()
    pyperclip.copy(text)
    time.sleep(0.05)
    kb = pykbd.Controller()
    with kb.pressed(pykbd.Key.ctrl):
        kb.press("v")
        kb.release("v")
    time.sleep(0.05)
    pyperclip.copy(original)
