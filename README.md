# TTS Dataset Processor

## Installation & Usage

> **Prerequisites:** This tool requires `ffmpeg` and `libc++1`. On Debian/Ubuntu, you can install them with:
>
> ```bash
> apt update && apt install ffmpeg libc++1 -y
> ```

### With Poetry (Recommended)

1.  **Install:**
    ```bash
    poetry install
    ```
2.  **Run:**
    ```bash
    poetry run python -m tts_dataset_processor.cli -i path/to/your/audio.mp3
    ```

4. update deps if changed:
    ```bash
    poetry update
    ```


**Note:** To install Poetry on Debian/Ubuntu, you can use `pipx`:
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
    Make sure your Python environment is activated, then run the script as a module:
    ```bash
    python -m tts_dataset_processor.cli -i path/to/your/audio.mp3
    ```

