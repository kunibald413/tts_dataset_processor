import argparse
from .main import run_pipeline


def main():
    """
    Command-line interface for the TTS Dataset Processor.
    """
    parser = argparse.ArgumentParser(
        description="Process a raw audio file (or a directory of them) into a TTS dataset."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input audio file or directory to process.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Path to the output directory. Defaults to 'output'.",
    )
    args = parser.parse_args()

    run_pipeline(input_path=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
