import json
import os
from dataclasses import dataclass
from typing import Optional, List

# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/results.html
'''
the manifest file should be a json file where each line has the following format:
{
   "audio_filepath": "/path/to/audio.wav",  # path to the audio file
   "duration": None,  # duration of the audio in seconds, set to `None` to use full audio
   "taskname": "asr",  # use "ast" for speech-to-text translation
   "source_lang": "en",  # language of the audio input, set `source_lang`==`target_lang` for ASR
   "target_lang": "en",  # language of the text output
   "pnc": "yes",  # whether to have PnC output, choices=['yes', 'no']
   "answer": "na", # set to non-dummy strings to calculate WER/BLEU scores
}
'''
@dataclass
class CanaryManifestItem:
    audio_filepath: str  # Path to the audio file
    duration: Optional[float] = None  # Duration of the audio in seconds
    taskname: str = "asr"  # Task name, default is "asr"

    # 'en', 'de', 'es', 'fr'
    source_lang: str = "en"  # Language of the audio input
    target_lang: str = "en"  # Language of the text output

    pnc: str = "yes"  # PnC output, choices=['yes', 'no']
    answer: str = "na"  # Non-dummy strings for calculating WER/BLEU scores


def create_manifest(
    audio_files,
    manifest_path,
    taskname="asr",
    source_lang="en",
    target_lang="en",
    pnc="yes",
    answer="na"
):
    """
    Creates a manifest file in JSON format where each line represents a dictionary
    containing metadata about the audio files.

    Parameters
    ----------
    audio_files : list of str
        List of paths to audio files.

    manifest_path : str
        Path to save the manifest file.

    taskname : str, optional
        Task name (e.g., 'asr' for automatic speech recognition or 'ast' for translation). Default is 'asr'.

    source_lang : str, optional
        Source language code. Default is 'en'.

    target_lang : str, optional
        Target language code. Default is 'en'.

    pnc : str, optional
        Whether to have PnC output. Choices are 'yes' or 'no'. Default is 'yes'.

    answer : str, optional
        Placeholder answer, used to calculate WER/BLEU scores. Default is 'na'.
    """
    manifest_data = []

    for audio_file in audio_files:
        if os.path.exists(audio_file):
            # Create a manifest entry for each audio file
            entry = CanaryManifestItem(
                audio_filepath=audio_file,
                duration=None,  # Set to None for full audio duration
                taskname=taskname,
                source_lang=source_lang,
                target_lang=target_lang,
                pnc=pnc,
                answer=answer
            )
            manifest_data.append(entry)
        else:
            print(f"File not found: {audio_file}")

    # Write manifest data to file, one JSON object per line
    with open(manifest_path, 'w') as f:
        for entry in manifest_data:
            json.dump(entry.__dict__, f)
            f.write('\n')

    print(f"Manifest file created at: {manifest_path}")

    return manifest_path

def read_manifest(manifest_path: str) -> List[CanaryManifestItem]:
    items = []
    with open(manifest_path, 'r') as f:
        for line in f:
            entry_data = json.loads(line)
            item = CanaryManifestItem(**entry_data)  # Unpack the dictionary into the dataclass
            items.append(item)
    return items
