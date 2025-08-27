import os
import shutil
import librosa
import logging
import sys
from audio_separator.separator import Separator
from pydub import AudioSegment
from ten_vad import TenVad
from pathlib import Path
from typing import Dict, List, Iterable, Any

from .audio_utils import (
    convert_to_16k_mono,
    standardization,
    export_to_wav,
    load_audio_data,
)
from .vad import cut_audio_segments, detect_and_merge_speech_segments
from .asr.canary.chunked_infer import transcribe_audio_dir

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# --- Internal Helper Functions ---


def _setup_directories(temp_dir: str, output_dir: str) -> Dict[str, str]:
    """Cleans and creates temporary and output directories for a pipeline run."""
    logger.info("Setting up directories...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dirs = {
        "vad": os.path.join(temp_dir, "1_vad_segments"),
        "standardized": os.path.join(temp_dir, "2_standardized"),
        "separated": os.path.join(temp_dir, "3_separated"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    logger.info(f"Temporary directory: {temp_dir}")
    logger.info(f"Output directory: {output_dir}")
    return dirs


def _run_vad_segmentation(
    input_file: str, vad_segments_dir: str, min_duration: float, max_duration: float
) -> List[str]:
    """Runs VAD on the input file and saves segments to a directory."""
    logger.info("[Step 1/5] Running VAD segmentation...")
    sr_vad, data_vad = convert_to_16k_mono(input_file)
    if data_vad is None:
        logger.error("VAD pre-processing failed, cannot continue.")
        return []

    ten_vad_instance = TenVad(256, 0.5)
    speech_timestamps = detect_and_merge_speech_segments(
        ten_vad_instance, data_vad, sr_vad, ten_vad_instance.hop_size, max_duration_s=max_duration
    )
    cut_audio_segments(speech_timestamps, input_file, vad_segments_dir, min_duration)

    segment_files = sorted(os.listdir(vad_segments_dir))
    logger.info(f"VAD produced {len(segment_files)} segments.")
    return [os.path.join(vad_segments_dir, f) for f in segment_files]


def _transcribe_segment(vocals_path: str, temp_dir: str) -> List[Dict]:
    """Transcribes a single audio file using the Canary ASR model."""
    logger.info("[Step 4/5] Transcribing vocals...")
    asr_input_dir = os.path.join(temp_dir, "4_asr_input")  # Subdir for this task
    if os.path.exists(asr_input_dir):
        shutil.rmtree(asr_input_dir)
    os.makedirs(asr_input_dir)

    shutil.copy(vocals_path, asr_input_dir)

    try:
        # transcribe_audio_dir returns a list of TranscriptionResult objects
        asr_results = transcribe_audio_dir(inp_audio_dir=asr_input_dir, result_to_file=False, lang="en")
        if not asr_results:
            logger.warning("ASR returned no transcription.")
            return []

        # The result is a list of dataclass objects, not dicts
        transcribed_text = asr_results[0].text
        duration = librosa.get_duration(path=vocals_path)
        logger.info(f"Transcription successful: '{transcribed_text[:50]}...'")
        return [{"start": 0, "end": duration, "text": transcribed_text}]

    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return []


def _process_segment(
    segment_path: str,
    dirs: Dict[str, str],
    separator: Separator,
    output_dir: str,
):
    """Runs steps 2-5 of the pipeline for a single audio segment."""
    segment_filename = os.path.basename(segment_path)
    base_temp_dir = os.path.dirname(dirs["vad"])

    # -- Step 2: Standardize --
    logger.info("[Step 2/5] Standardizing segment...")
    standardized_data = standardization(segment_path)
    # Separator needs a file path, so we save the standardized audio temporarily.
    temp_export_prefix = os.path.splitext(segment_filename)[0]
    export_to_wav(
        audio_data=standardized_data,
        asr_result=[{"start": 0, "end": standardized_data.waveform.shape[0] / standardized_data.sample_rate}],
        folder_path=dirs["standardized"],
        file_name_prefix=temp_export_prefix,
    )
    standardized_file_path = next(Path(dirs["standardized"]).glob(f"{temp_export_prefix}*.wav"))
    logger.info(f"Standardized segment saved to: {standardized_file_path}")

    # -- Step 3: Separate Vocals/Instruments --
    logger.info("[Step 3/5] Separating vocals from instruments...")
    try:
        # The separator returns a list of filenames, not full paths.
        output_filenames = separator.separate([standardized_file_path])
        vocals_filename = next((f for f in output_filenames if "(vocals)" in f), None)
        
        if not vocals_filename:
            logger.warning("Could not find vocals file in separator output. Skipping segment.")
            return

        # Construct the full, absolute path and verify it exists.
        vocals_path = os.path.abspath(os.path.join(separator.output_dir, vocals_filename))
        if not os.path.exists(vocals_path):
            logger.error(f"Separated vocals file not found at expected path: {vocals_path}. Skipping segment.")
            return

        logger.info(f"Vocals separated to: {vocals_path}")

    except Exception as e:
        logger.error(f"Error during separation: {e}. Skipping segment.", exc_info=True)
        return

    # -- Step 4: Transcribe (ASR) --
    asr_result = _transcribe_segment(vocals_path, base_temp_dir)
    if not asr_result:
        logger.warning("Transcription failed. Skipping segment.")
        return

    # -- Step 5: Export Final Audio --
    logger.info("[Step 5/5] Exporting final audio segment...")
    vocals_audio_data = standardization(vocals_path)  # Re-load vocals to get AudioData
    final_filename_prefix = os.path.splitext(segment_filename)[0]
    export_to_wav(
        audio_data=vocals_audio_data,
        asr_result=asr_result,
        folder_path=output_dir,
        file_name_prefix=final_filename_prefix,
    )
    logger.info(f"Final segment exported to '{output_dir}' with prefix '{final_filename_prefix}'.")


def _group_into_chunks(data: Iterable[Any], chunk_size: int) -> Iterable[List[Any]]:
    """Groups an iterable into chunks of a specific size."""
    data = list(data)
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def _batch_standardize(vad_segment_paths: List[str], standardized_dir: str) -> List[Path]:
    """Runs standardization on a list of audio files."""
    logger.info("\n--- [Step 2/5] Standardizing all segments ---")
    standardized_files = []
    for segment_path in vad_segment_paths:
        segment_filename = os.path.basename(segment_path)
        standardized_data = standardization(segment_path)
        temp_export_prefix = os.path.splitext(segment_filename)[0]
        # Temporarily save the standardized file for the next step
        export_to_wav(
            audio_data=standardized_data,
            asr_result=[{"start": 0, "end": standardized_data.waveform.shape[0] / standardized_data.sample_rate}],
            folder_path=standardized_dir,
            file_name_prefix=temp_export_prefix,
        )
        standardized_file_path = next(Path(standardized_dir).glob(f"{temp_export_prefix}*.wav"))
        standardized_files.append(standardized_file_path)
        logger.info(f"  - Standardized: {standardized_file_path}")
    return standardized_files


def _batch_separate(standardized_files: List[Path], separator: Separator, temp_dir: str) -> List[str]:
    """
    Runs vocal separation on a list of standardized audio files,
    grouping short files together to improve separation quality.
    """
    logger.info("\n--- [Step 3/5] Separating vocals for all segments (with chunking) ---")
    
    # Create a temporary directory for the concatenated audio chunks
    concatenated_dir = os.path.join(temp_dir, "3a_concatenated")
    os.makedirs(concatenated_dir, exist_ok=True)

    vocal_files = []
    file_chunks = _group_into_chunks(standardized_files, 4)

    for i, chunk in enumerate(file_chunks):
        if not chunk:
            continue

        logger.info(f"  - Processing chunk {i+1} with {len(chunk)} files...")
        
        # --- Concatenate files in the chunk ---
        combined_audio = AudioSegment.empty()
        durations_ms = []
        original_filenames = []

        for file_path in chunk:
            audio_segment = AudioSegment.from_file(file_path)
            combined_audio += audio_segment
            durations_ms.append(len(audio_segment))
            original_filenames.append(os.path.basename(file_path))

        concatenated_path = os.path.join(concatenated_dir, f"chunk_{i}.wav")
        combined_audio.export(concatenated_path, format="wav")

        # --- Run separator on the single concatenated file ---
        try:
            output_filenames = separator.separate([concatenated_path])
            vocals_filename = next((f for f in output_filenames if "(vocals)" in f), None)
            
            if not vocals_filename:
                logger.warning(f"No vocals file found for chunk {i}. Skipping.")
                continue

            vocals_path = os.path.abspath(os.path.join(separator.output_dir, vocals_filename))
            if not os.path.exists(vocals_path):
                logger.error(f"Separated vocals file for chunk {i} not found. Skipping.")
                continue

            # --- Split the separated vocals back into individual files ---
            separated_vocals_audio = AudioSegment.from_file(vocals_path)
            start_ms = 0
            for j, duration_ms in enumerate(durations_ms):
                end_ms = start_ms + duration_ms
                split_vocal = separated_vocals_audio[start_ms:end_ms]
                
                # Create a new filename for the split vocal file
                original_base_name = os.path.splitext(original_filenames[j])[0]
                split_vocal_filename = f"{original_base_name}_(vocals).wav"
                split_vocal_path = os.path.join(separator.output_dir, split_vocal_filename)
                
                split_vocal.export(split_vocal_path, format="wav")
                vocal_files.append(split_vocal_path)
                logger.info(f"    - Split and saved separated vocal to: {split_vocal_path}")
                
                start_ms = end_ms

        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}", exc_info=True)

    return vocal_files


def _batch_transcribe(vocal_files: List[str], temp_dir: str) -> Dict[str, List[Dict]]:
    """Runs transcription on a batch of vocal files."""
    logger.info("\n--- [Step 4/5] Transcribing all vocal segments in a single batch ---")
    asr_input_dir = os.path.join(temp_dir, "4_asr_input")
    if os.path.exists(asr_input_dir):
        shutil.rmtree(asr_input_dir)
    os.makedirs(asr_input_dir)

    logger.info(f"Copying {len(vocal_files)} vocal files to a temporary directory for ASR...")
    for vocal_path in vocal_files:
        shutil.copy(vocal_path, asr_input_dir)

    try:
        # Run transcription on the entire directory at once
        asr_results = transcribe_audio_dir(inp_audio_dir=asr_input_dir, result_to_file=False, lang="en")
        
        # Create a dictionary mapping the original filepath to its transcription
        transcriptions = {}
        for result in asr_results:
            # The result.filepath is inside the asr_input_dir, so we map it back
            # to the full original path by matching the basename.
            full_original_path = next((p for p in vocal_files if os.path.basename(p) == os.path.basename(result.filepath)), None)

            if full_original_path:
                duration = librosa.get_duration(path=full_original_path)
                transcriptions[full_original_path] = [{"start": 0, "end": duration, "text": result.text}]
                logger.info(f"  - Transcribed '{os.path.basename(full_original_path)}': '{result.text[:50]}...'")

        return transcriptions

    except Exception as e:
        logger.error(f"Error during batch transcription: {e}", exc_info=True)
        return {}


def _batch_export(transcriptions: Dict[str, List[Dict]], output_dir: str):
    """Runs final export for a dictionary of transcribed audio files."""
    logger.info("\n--- [Step 5/5] Exporting all final segments ---")
    for vocal_path, asr_result in transcriptions.items():
        if not asr_result:
            logger.warning(f"No transcription result for {vocal_path}. Skipping export.")
            continue

        vocals_audio_data = load_audio_data(vocal_path)
        base_filename = os.path.basename(vocal_path).replace("_(vocals)", "")
        final_filename_prefix = os.path.splitext(base_filename)[0]
        export_to_wav(
            audio_data=vocals_audio_data,
            asr_result=asr_result,
            folder_path=output_dir,
            file_name_prefix=final_filename_prefix,
        )
        logger.info(f"  - Final segment exported to '{output_dir}' with prefix '{final_filename_prefix}'.")


# --- Main Pipeline Runner ---


def run_pipeline(
    input_file: str,
    temp_dir: str = "tmp",
    output_dir: str = "output",
    min_duration: float = 2.5,
    max_duration: float = 11.5,
    separator_model_file_name: str = "melband_roformer_big_beta4.ckpt",
):
    """
    Runs the full audio processing pipeline in batch stages.
    1. VAD Segmentation -> 2. Standardization -> 3. Separation -> 4. ASR -> 5. Export
    """
    logger.info(f"--- Starting Pipeline for: {input_file} ---")

    # -- Setup --
    dirs = _setup_directories(temp_dir, output_dir)

    # -- Step 1: VAD Segmentation --
    vad_segment_paths = _run_vad_segmentation(
        input_file, dirs["vad"], min_duration, max_duration
    )
    if not vad_segment_paths:
        logger.error("No segments produced by VAD. Halting pipeline.")
        return

    # -- Step 2: Batch Standardization --
    standardized_files = _batch_standardize(vad_segment_paths, dirs["standardized"])

    # -- Step 3: Batch Separation --
    separator = Separator(output_dir=dirs["separated"])
    separator.load_model(separator_model_file_name)
    vocal_files = _batch_separate(standardized_files, separator, temp_dir)

    # -- Step 4: Batch Transcription --
    transcriptions = _batch_transcribe(vocal_files, temp_dir)

    # -- Step 5: Batch Export --
    _batch_export(transcriptions, output_dir)

    logger.info("\n--- Pipeline Finished ---")


# --- Main Entry Point ---

if __name__ == "__main__":
    input_audio = r"E:\tts\xtts\Knut\_raw_data\0\Knut_Im_back.mp3"
    run_pipeline(input_audio)