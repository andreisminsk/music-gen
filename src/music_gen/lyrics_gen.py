"""Generate song lyrics using a local LLM via Ollama.

Produces lyrics with [Verse], [Chorus], etc. markers that YuE2 expects.
"""

import json
import sys
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MODEL = "gemma4:31b-cloud"

SYSTEM_PROMPT = """\
You are a professional songwriter. Write song lyrics that are creative, \
rhythmic, and emotionally resonant.

RULES:
- Use section markers: [Verse 1], [Verse 2], [Chorus], [Bridge], [Outro]
- Each section should have 2-4 lines
- Include a chorus that repeats (appear at least twice)
- Match the mood and genre of the requested style
- Write in the requested language
- Do NOT add any commentary, explanations, or metadata — only the lyrics
- Do NOT wrap output in code blocks or quotes"""

STYLE_HINTS = {
    "rock": "Use vivid imagery, strong rhythm, and emotional intensity.",
    "pop": "Keep it catchy, relatable, with simple but memorable hooks.",
    "jazz": "Use sophisticated wordplay, introspective themes, and smooth flow.",
    "metal": "Use dark imagery, powerful metaphors, and aggressive energy.",
    "folk": "Tell a story, use nature imagery, and keep language grounded.",
    "hip-hop": "Use wordplay, internal rhymes, and rhythmic cadence.",
    "electronic": "Use futuristic imagery, repetition, and hypnotic phrases.",
    "ballad": "Focus on deep emotion, vulnerability, and narrative arc.",
    "disco": "Use fun, danceable themes with repetitive hooks.",
    "blues": "Use call-and-response, hardship themes, and raw emotion.",
}


def _pick_style_hint(style: str) -> str:
    """Add a genre-specific hint based on the style prompt."""
    style_lower = style.lower()
    for genre, hint in STYLE_HINTS.items():
        if genre in style_lower:
            return hint
    return "Write with authentic emotion and memorable phrasing."


def generate_lyrics(
    style: str,
    topic: str = "",
    language: str = "English",
    model: str = DEFAULT_MODEL,
    n_sections: int = 4,
    temperature: float = 0.8,
) -> str:
    """Generate song lyrics via Ollama.

    Args:
        style: Style/genre description (e.g. "Russian rock, powerful male vocal").
        topic: Optional topic or story for the song.
        language: Language to write in (default: English).
        model: Ollama model name.
        n_sections: Approximate number of sections (default: 4).
        temperature: Sampling temperature (default: 0.8).

    Returns:
        Generated lyrics text with section markers.
    """
    import ollama

    hint = _pick_style_hint(style)
    topic_line = f"\nTopic/story: {topic}" if topic else ""

    user_prompt = (
        f"Write song lyrics in {language}.\n"
        f"Style/genre: {style}\n"
        f"Sections: ~{n_sections} (include at least one chorus that repeats)\n"
        f"{hint}{topic_line}"
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
    )

    lyrics = response["message"]["content"].strip()

    # Strip markdown code fences if the model added them
    if lyrics.startswith("```"):
        lines = lyrics.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        lyrics = "\n".join(lines).strip()

    return lyrics


def save_lyrics(lyrics: str, path: str) -> str:
    """Save lyrics to a text file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(lyrics, encoding="utf-8")
    return str(out)


def generate_and_save(
    style: str,
    output: str = "lyrics/song.txt",
    topic: str = "",
    language: str = "English",
    model: str = DEFAULT_MODEL,
    n_sections: int = 4,
    temperature: float = 0.8,
) -> str:
    """Generate lyrics and save to file. Returns the file path."""
    lyrics = generate_lyrics(
        style=style, topic=topic, language=language,
        model=model, n_sections=n_sections, temperature=temperature,
    )
    path = save_lyrics(lyrics, output)
    return path
