import argparse
from .main import run_pipeline
from .transcribe_utils import transcribe_files


def main():
    """
    Command-line interface for the TTS Dataset Processor.
    """
    parser = argparse.ArgumentParser(
        description="TTS Dataset Processor - Process audio into TTS datasets or transcribe files"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full TTS dataset processing pipeline')
    pipeline_parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to the input audio file or directory to process"
    )
    pipeline_parser.add_argument(
        "-o", "--output", type=str, default="output",
        help="Path to the output directory. Defaults to 'output'"
    )
    
    # Transcribe-only command
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe audio files only')
    transcribe_parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to audio file or directory to transcribe"
    )
    transcribe_parser.add_argument(
        "--model", type=str, default="nvidia/canary-1b-flash",
        help="ASR model to use for transcription"
    )
    transcribe_parser.add_argument(
        "--lang", type=str, default="en",
        help="Language code for transcription"
    )
    transcribe_parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip creating CSV output file"
    )
    transcribe_parser.add_argument(
        "--no-json", action="store_true", 
        help="Skip creating JSON output file"
    )
    
    args = parser.parse_args()
    
    if args.command == 'pipeline':
        run_pipeline(input_path=args.input, output_dir=args.output)
    elif args.command == 'transcribe':
        transcribe_files(
            input_path=args.input,
            asr_model_name=args.model,
            lang=args.lang,
            output_csv=not args.no_csv,
            output_json=not args.no_json
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
