
# from: https://github.com/NVIDIA/NeMo/commit/0bb9e66a6d29b28e8831d1d1dd8a30310173ce46

import os
from pathlib import Path

from typing import List

from .speech_to_text_aed_chunked_infer import main as canary_chunked_asr
from .speech_to_text_aed_chunked_infer import TranscriptionConfig, TranscriptionResult
from .canary_manifest_utils import create_manifest

CANARY_LANGS = {'en', 'de', 'es', 'fr'}

def transcribe_audio_dir(inp_audio_dir: str, result_to_file: bool = False, lang: str = 'en') -> List[TranscriptionResult]:
    cfg = TranscriptionConfig()
    cfg.pretrained_name = "nvidia/canary-1b"

    if not is_valid_lang(lang):
        raise Exception(f"invalid canary language: '{lang}', must be one of {CANARY_LANGS}")

    if lang == 'en':
        cfg.audio_dir = inp_audio_dir
    else:
        path_inp_audio_dir = Path(inp_audio_dir)
        manifest_out_path = path_inp_audio_dir / 'canary_manifest.json'
        audio_files = list(path_inp_audio_dir.rglob('*.wav')) + list(path_inp_audio_dir.rglob('*.mp3'))
        audio_files = [a.as_posix() for a in audio_files]
        cfg.dataset_manifest = create_manifest(audio_files, manifest_out_path.as_posix(), source_lang=lang, target_lang=lang)

    cfg.write_transcription_to_file = result_to_file
    cfg, results = canary_chunked_asr(cfg)

    return results


def is_valid_lang(lang: str):
    return lang in CANARY_LANGS


if __name__ == '__main__':
    transcribe_audio_dir("")
