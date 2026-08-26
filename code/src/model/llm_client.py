"""Shared OpenAI-compatible client helpers.

Credentials are read at call time so CLI compatibility flags can populate the
environment after imports without keeping secrets in source files.
"""

import os
import threading

import openai


CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
_thread_clients = threading.local()


def _api_keys():
    keys = [key.strip() for key in os.getenv("DREAMS_OPENAI_API_KEYS", "").split(",") if key.strip()]
    primary_key = os.getenv("OPENAI_API_KEY", "").strip()
    if primary_key and primary_key not in keys:
        keys.insert(0, primary_key)
    if not keys:
        raise RuntimeError("Set OPENAI_API_KEY before running DREAMS.")
    return keys


def get_client(key_index=0):
    keys = _api_keys()
    api_key = keys[key_index % len(keys)]
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    cache = getattr(_thread_clients, "cache", None)
    if cache is None:
        cache = _thread_clients.cache = {}
    cache_key = (api_key, base_url)
    if cache_key not in cache:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        cache[cache_key] = openai.OpenAI(**kwargs)
    return cache[cache_key]


def call_embedding(prompt):
    return get_client().embeddings.create(model=EMBEDDING_MODEL, input=prompt)


def _call_chat(messages, model_name, temperature, seed, json_mode, key_index):
    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    completion = get_client(key_index).chat.completions.create(**kwargs)
    return completion.choices[0].message.content


def call_chatgpt_1(messages, model_name, temperature, seed, json_mode):
    return _call_chat(messages, model_name, temperature, seed, json_mode, 0)


def call_chatgpt_2(messages, model_name, temperature, seed, json_mode):
    return _call_chat(messages, model_name, temperature, seed, json_mode, 1)


def call_chatgpt_3(messages, model_name, temperature, seed, json_mode):
    return _call_chat(messages, model_name, temperature, seed, json_mode, 2)
