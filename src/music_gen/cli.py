"""Command-line interface for YuE2 music generation."""

import argparse
import logging
import sys

from .pipeline import load_pipeline, generate_song, load_prompt_from_json
from .lyrics_gen import generate_lyrics, save_lyrics, DEFAULT_MODEL as LYRICS_DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="music-gen",
        description="Generate music with YuE2-3B",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = sub.add_parser("generate", help="Generate a song from style + lyrics")
    gen.add_argument("--style", "-s", help='Style prompt, e.g. "Jazz, warm vocal, piano"')
    gen.add_argument("--lyrics", "-l", help="Lyrics text (use \\n for line breaks)")
    gen.add_argument("--lyrics-file", "-L", help="Path to a text file with lyrics")
    gen.add_argument(
        "--prompt-json", "-p",
        help="Path to prompt JSON file (keys: style, lyrics, seed)",
    )
    gen.add_argument(
        "--cot", choices=["full", "melody", "off"], default="full",
        help="Chain-of-thought mode (default: full)",
    )
    gen.add_argument("--seed", type=int, default=None, help="Random seed")
    gen.add_argument("--cfg-scale", type=float, default=1.2, help="CFG scale (default: 1.2)")
    gen.add_argument("--abc", default=None, help="Path to ABC notation file for covers/edits")
    gen.add_argument("--output-dir", "-o", default="output", help="Output directory")
    gen.add_argument("--filename", "-f", default="song", help="Output filename (no extension)")
    gen.add_argument("--model", default="m-a-p/YuE2-3B", help="Model repo ID")
    gen.add_argument(
        "--vae", default="m-a-p/YuE2-Vae",
        help="VAE repo (m-a-p/YuE2-Vae or m-a-p/YuE2-Vae-legacy)",
    )
    gen.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    gen.add_argument("--no-progress", action="store_true", help="Suppress progress messages")
    gen.add_argument("--generate-lyrics", action="store_true",
                     help="Generate lyrics via Ollama before composing")
    gen.add_argument("--topic", "-t", default="", help="Topic for lyrics generation")
    gen.add_argument("--language", default="English", help="Language for lyrics generation (default: English)")
    gen.add_argument("--lyrics-model", default="gemma4:31b-cloud", help="Ollama model for lyrics (default: gemma4:31b-cloud)")
    gen.add_argument("--lyrics-output", default=None,
                     help="Save generated lyrics to this file (default: lyrics/generated.txt)")

    # --- plan ---
    plan_cmd = sub.add_parser("plan", help="Export a symbolic plan (ABC) without generating audio")
    plan_cmd.add_argument("--style", "-s", required=True, help="Style prompt")
    plan_cmd.add_argument("--lyrics", "-l", required=True, help="Lyrics text")
    plan_cmd.add_argument("--cot", choices=["full", "melody"], default="full")
    plan_cmd.add_argument("--seed", type=int, default=None)
    plan_cmd.add_argument("--output-dir", "-o", default="output")
    plan_cmd.add_argument("--model", default="m-a-p/YuE2-3B")
    plan_cmd.add_argument("--vae", default="m-a-p/YuE2-Vae")
    plan_cmd.add_argument("--device", default="cuda")

    # --- lyrics-gen ---
    lyr = sub.add_parser("lyrics-gen", help="Generate lyrics using a local LLM (Ollama)")
    lyr.add_argument("--style", "-s", required=True, help='Style/genre, e.g. "Russian rock, powerful vocal"')
    lyr.add_argument("--topic", "-t", default="", help="Optional topic or story for the song")
    lyr.add_argument("--language", "-L", default="English", help="Language to write in (default: English)")
    lyr.add_argument("--model", "-m", default=LYRICS_DEFAULT_MODEL, help=f"Ollama model (default: {LYRICS_DEFAULT_MODEL})")
    lyr.add_argument("--sections", "-n", type=int, default=4, help="Approximate number of sections (default: 4)")
    lyr.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    lyr.add_argument("--output", "-o", default="lyrics/song.txt", help="Output file path (default: lyrics/song.txt)")
    lyr.add_argument("--print-only", action="store_true", help="Print lyrics to stdout only, don't save to file")

    args = parser.parse_args(argv)

    if args.command == "generate":
        style = args.style
        lyrics = args.lyrics
        seed = args.seed

        if args.prompt_json:
            data = load_prompt_from_json(args.prompt_json)
            style = style or data.get("style")
            lyrics = lyrics or data.get("lyrics")
            seed = seed or data.get("seed")

        if args.lyrics_file:
            from pathlib import Path
            lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")

        # Generate lyrics via Ollama if requested
        if args.generate_lyrics:
            from .lyrics_gen import generate_lyrics, save_lyrics
            print(f"✍️  Generating lyrics (language: {args.language})...")
            lyrics = generate_lyrics(
                style=style or "Pop",
                topic=args.topic,
                language=args.language,
                model=args.lyrics_model,
            )
            print(f"📝 Generated lyrics:\n{lyrics}\n")
            lyrics_path = args.lyrics_output or "lyrics/generated.txt"
            save_lyrics(lyrics, lyrics_path)
            print(f"💾 Lyrics saved to: {lyrics_path}")

        if not style or not lyrics:
            parser.error("Provide --style and --lyrics/--lyrics-file, or --generate-lyrics")

        abc_text = None
        if args.abc:
            from pathlib import Path
            abc_text = Path(args.abc).read_text(encoding="utf-8")

        pipe = load_pipeline(
            model_id=args.model,
            vae=args.vae,
            device=args.device,
            progress=not args.no_progress,
        )
        result = generate_song(
            pipe=pipe,
            style=style,
            lyrics=lyrics,
            cot=args.cot,
            seed=seed,
            cfg_scale=args.cfg_scale,
            abc=abc_text,
            output_dir=args.output_dir,
            filename=args.filename,
        )
        print(f"\n✅ Song saved to: {result}")

        pipe.close()

    elif args.command == "plan":
        pipe = load_pipeline(model_id=args.model, vae=args.vae, device=args.device)
        plan = pipe.plan(style=args.style, lyrics=args.lyrics, cot=args.cot, seed=args.seed)
        from pathlib import Path
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plan.save(str(out / "plan"))
        print(f"\n✅ Plan saved to: {out / 'plan'}")
        pipe.close()

    elif args.command == "lyrics-gen":
        from .lyrics_gen import generate_lyrics
        print(f"🎵 Generating lyrics (style: {args.style}, language: {args.language})...")
        lyrics = generate_lyrics(
            style=args.style,
            topic=args.topic,
            language=args.language,
            model=args.model,
            n_sections=args.sections,
            temperature=args.temperature,
        )
        if args.print_only:
            print("\n" + lyrics)
        else:
            path = save_lyrics(lyrics, args.output)
            print(f"\n✅ Lyrics saved to: {path}")
            print(f"\n--- Lyrics preview ---\n{lyrics[:500]}")
            if len(lyrics) > 500:
                print(f"... ({len(lyrics)} chars total)")


if __name__ == "__main__":
    main()
