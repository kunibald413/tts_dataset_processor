# TTS Dataset Processor

## Installation & Usage

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

**Note:** To install Poetry on Debian/Ubuntu, you can use `pipx`:
```bash
sudo apt update
sudo apt install pipx -y
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

