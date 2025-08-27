import argparse
from .main import run_pipeline


def main():
    """
    Command-line interface for the TTS Dataset Processor.
    """
    parser = argparse.ArgumentParser(
        description="Process a raw audio file into a TTS dataset."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input audio file to process.",
    )
    args = parser.parse_args()

    run_pipeline(input_file=args.input)


if __name__ == "__main__":
    main()
