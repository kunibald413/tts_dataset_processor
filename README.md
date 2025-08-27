# TTS Dataset Processor

## Installation & Usage

> **Prerequisite:** This tool requires `ffmpeg`. On Debian/Ubuntu, you can install it with:
>
> ```bash
> apt update && apt install ffmpeg -y
> ```

### With Poetry (Recommended)

1.  **Install:**
    ```bash
    poetry install
    ```
2.  **Configure:** Edit `src/tts_dataset_processor/main.py` to set the `input_audio` file path.
3.  **Run:**
    ```bash
    poetry run python -m tts_dataset_processor.main
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
2.  **Configure:** Edit `src/tts_dataset_processor/main.py` to set the `input_audio` file path.
3.  **Run:**
    Make sure your Python environment is activated, then run the script as a module:
    ```bash
    python -m tts_dataset_processor.main
    ```

