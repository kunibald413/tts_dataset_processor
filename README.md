# TTS Dataset Processor

Process audio files into TTS-ready datasets or transcribe files standalone.

## Installation & Usage

> **Prerequisites:** Requires `ffmpeg` and `libc++1`:
> ```bash
> apt update && apt install ffmpeg libc++1 -y
> ```

### With Poetry (Recommended)

1.  **Install:**
    ```bash
    poetry install
    ```
2.  **Run Pipeline:**
    ```bash
    poetry run python -m tts_dataset_processor pipeline -i path/to/audio.mp3
    ```
3.  **Transcribe Only:**
    ```bash
    poetry run python -m tts_dataset_processor transcribe -i /ref_audios/reference_audios
    ```

**Note:** To install Poetry on Debian/Ubuntu:
```bash
apt update
apt install pipx -y
pipx ensurepath
pipx install poetry
```

### Without Poetry

1.  **Install:**
    ```bash
    pip install .
    ```
2.  **Run:**
    ```bash
    # Full TTS dataset pipeline
    python -m tts_dataset_processor pipeline -i audio.mp3 -o output

    # Transcribe files only (creates CSV + JSON)
    python -m tts_dataset_processor transcribe -i audio_folder
    ```

## Commands

- `pipeline` - Full TTS dataset processing (VAD, separation, transcription, export)
- `transcribe` - Standalone transcription with metadata export

