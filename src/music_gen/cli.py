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
    gen.add_argument("--melody-only", action="store_true",
                     help="Export only the melody from the plan (ABC + MIDI), no full song audio")
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

    # --- batch ---
    bat = sub.add_parser(
        "batch",
        help="Run multiple jobs from a zip or directory containing job.json",
    )
    bat_src = bat.add_mutually_exclusive_group(required=True)
    bat_src.add_argument("--zip", dest="zip_path", help="Path to a batch zip archive")
    bat_src.add_argument("--dir", dest="dir_path", help="Path to a directory containing job.json")
    bat.add_argument("--output-dir", "-o", default=None,
                     help="Override the batch-level output directory")
    bat.add_argument("--model", default=None, help="Override the batch-level model repo ID")
    bat.add_argument("--vae", default=None, help="Override the batch-level VAE repo")
    bat.add_argument("--device", default=None, help="Override the batch-level torch device")
    bat.add_argument("--stop-on-error", action="store_true",
                     help="Abort the batch on the first job failure")
    bat.add_argument("--report", default=None,
                     help="Batch report JSON path (default: <output-dir>/batch_report.json)")
    bat.add_argument("--no-progress", action="store_true", help="Suppress progress messages")

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

        # Melody-only mode: export plan + melody, skip audio generation
        if args.melody_only:
            from pathlib import Path
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            plan = pipe.plan(style=style, lyrics=lyrics, cot=args.cot, seed=seed)
            plan.save(str(out / "melody_plan"))
            print(f"\n✅ Melody plan saved to: {out / 'melody_plan'}")
            print(f"   ABC score: {out / 'melody_plan' / 'score.abc'}")
            pipe.close()
            return

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

    elif args.command == "batch":
        from .batch import BatchValidationError, load_batch, run_batch

        try:
            spec = load_batch(args.zip_path or args.dir_path)
        except BatchValidationError as e:
            print("❌ Batch validation failed:", file=sys.stderr)
            for err in e.errors:
                print(f"  - {err}", file=sys.stderr)
            return 2

        # CLI flags override batch-level defaults
        for key, val in (("model", args.model), ("vae", args.vae), ("device", args.device)):
            if val is not None:
                spec.defaults[key] = val
        if args.output_dir is not None:
            spec.defaults["output_dir"] = args.output_dir

        logger.info("Batch: %d job(s) from %s → %s",
                    len(spec.jobs), spec.source, spec.defaults["output_dir"])
        pipe = load_pipeline(
            model_id=spec.defaults["model"],
            vae=spec.defaults["vae"],
            device=spec.defaults["device"],
            progress=not args.no_progress,
        )
        try:
            report = run_batch(
                spec,
                pipe,
                output_dir=args.output_dir,
                stop_on_error=args.stop_on_error,
                report_path=args.report,
            )
        except KeyboardInterrupt:
            print("\n⚠️  Batch interrupted — partial report saved.", file=sys.stderr)
            return 130
        finally:
            pipe.close()
            spec.cleanup()

        summary = report["summary"]
        print(f"\n✅ Batch finished: {summary['succeeded']}/{summary['total']} succeeded, "
              f"{summary['failed']} failed")
        return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    main()
