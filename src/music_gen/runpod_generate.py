#!/usr/bin/env python3
"""Quick-start script: generate music on RunPod after running deploy_runpod.sh.

Usage:
    conda activate yue2
    python -m music_gen.runpod_generate [--example jazz_funk|russian_rock|cyber_metal|mandarin_pop]
                                        [--style STYLE] [--lyrics LYRICS]
                                        [--seed SEED] [--cot full|melody|off]
                                        [--output-dir OUTPUT_DIR]
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

from music_gen.pipeline import load_pipeline, generate_song
from music_gen.examples import get_example, list_examples


def main():
    parser = argparse.ArgumentParser(description="Quick-start YuE2 generation on RunPod")
    parser.add_argument("--example", "-e", choices=list_examples(),
                        help="Use a built-in example prompt")
    parser.add_argument("--style", "-s", help="Style prompt (overrides example)")
    parser.add_argument("--lyrics", "-l", help="Lyrics text (overrides example)")
    parser.add_argument("--lyrics-file", "-L", help="Path to a text file with lyrics")
    parser.add_argument("--generate-lyrics", action="store_true",
                        help="Generate lyrics via Ollama before composing")
    parser.add_argument("--topic", "-t", default="", help="Topic for lyrics generation")
    parser.add_argument("--language", default="English", help="Language for lyrics generation")
    parser.add_argument("--lyrics-model", default="gemma4:31b-cloud", help="Ollama model for lyrics")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--cot", choices=["full", "melody", "off"], default="full",
                        help="Chain-of-thought mode")
    parser.add_argument("--cfg-scale", type=float, default=1.2, help="CFG scale")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Torch device")
    args = parser.parse_args()

    # Resolve prompt
    if args.example:
        prompt = get_example(args.example)
        style = args.style or prompt["style"]
        lyrics = args.lyrics or prompt["lyrics"]
        seed = args.seed or prompt.get("seed", 42)
    elif args.style and (args.lyrics or args.lyrics_file):
        style = args.style
        if args.lyrics_file:
            from pathlib import Path
            lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")
        else:
            lyrics = args.lyrics
        seed = args.seed or 42
    else:
        parser.error("Provide --example or both --style and --lyrics/--lyrics-file")

    # Generate lyrics if requested
    if args.generate_lyrics:
        from music_gen.lyrics_gen import generate_lyrics
        print(f"✍️  Generating lyrics (language: {args.language})...")
        lyrics = generate_lyrics(
            style=style, topic=args.topic, language=args.language, model=args.lyrics_model,
        )
        print(f"📝 Generated lyrics:\n{lyrics}\n")

    print(f"🎵 Style:   {style}")
    print(f"📝 Lyrics:  {lyrics[:80]}...")
    print(f"🎲 Seed:    {seed}  |  CoT: {args.cot}  |  CFG: {args.cfg_scale}")
    print()

    pipe = load_pipeline(device=args.device)
    result = generate_song(
        pipe, style=style, lyrics=lyrics, cot=args.cot,
        seed=seed, cfg_scale=args.cfg_scale, output_dir=args.output_dir,
    )
    print(f"\n✅ Song saved to: {result}")
    pipe.close()


if __name__ == "__main__":
    main()
