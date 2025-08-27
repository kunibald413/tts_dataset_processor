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
from dataclasses import dataclass

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


# --- Result Dataclasses ---


@dataclass
class FailedFile:
    """Represents a file that failed during a pipeline step."""

    filepath: str
    reason: str


@dataclass
class StepResult:
    """Holds the results of a pipeline step."""

    successful_outputs: Any
    failures: List[FailedFile]


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
) -> StepResult:
    """Runs VAD on the input file and saves segments to a directory."""
    logger.info("[Step 1/5] Running VAD segmentation...")
    try:
        sr_vad, data_vad = convert_to_16k_mono(input_file)
        if data_vad is None:
            raise ValueError("VAD pre-processing failed (could not load audio).")

        ten_vad_instance = TenVad(256, 0.5)
        speech_timestamps = detect_and_merge_speech_segments(
            ten_vad_instance, data_vad, sr_vad, ten_vad_instance.hop_size, max_duration_s=max_duration
        )
        cut_audio_segments(speech_timestamps, input_file, vad_segments_dir, min_duration)

        segment_files = sorted(os.listdir(vad_segments_dir))
        logger.info(f"VAD produced {len(segment_files)} segments.")
        output_paths = [os.path.join(vad_segments_dir, f) for f in segment_files]
        return StepResult(successful_outputs=output_paths, failures=[])

    except Exception as e:
        logger.error(f"VAD step failed entirely: {e}", exc_info=True)
        failure = FailedFile(filepath=input_file, reason=str(e))
        return StepResult(successful_outputs=[], failures=[failure])


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


# --- Batch Processing Functions ---


def _batch_standardize(vad_segment_paths: List[str], standardized_dir: str) -> StepResult:
    """Runs standardization on a list of audio files."""
    logger.info("\n--- [Step 2/5] Standardizing all segments ---")
    standardized_files = []
    failures = []
    for segment_path in vad_segment_paths:
        try:
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
        except Exception as e:
            logger.error(f"Failed to standardize {segment_path}: {e}", exc_info=True)
            failures.append(FailedFile(filepath=segment_path, reason=str(e)))

    return StepResult(successful_outputs=standardized_files, failures=failures)


def _batch_separate(standardized_files: List[Path], separator: Separator, temp_dir: str) -> StepResult:
    """
    Runs vocal separation on a list of standardized audio files,
    grouping short files together to improve separation quality.
    """
    logger.info("\n--- [Step 3/5] Separating vocals for all segments (with chunking) ---")
    
    # Create a temporary directory for the concatenated audio chunks
    concatenated_dir = os.path.join(temp_dir, "3a_concatenated")
    os.makedirs(concatenated_dir, exist_ok=True)

    vocal_files = []
    failures = []
    file_chunks = _group_into_chunks(standardized_files, 4)

    for i, chunk in enumerate(file_chunks):
        if not chunk:
            continue

        logger.info(f"  - Processing chunk {i+1} with {len(chunk)} files...")
        
        try:
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
            output_filenames = separator.separate([concatenated_path])
            vocals_filename = next((f for f in output_filenames if "(vocals)" in f), None)
            
            if not vocals_filename:
                raise RuntimeError("No vocals file found in separator output.")

            vocals_path = os.path.abspath(os.path.join(separator.output_dir, vocals_filename))
            if not os.path.exists(vocals_path):
                raise FileNotFoundError(f"Separated vocals file for chunk {i} not found.")

            # --- Split the separated vocals back into individual files ---
            separated_vocals_audio = AudioSegment.from_file(vocals_path)
            start_ms = 0
            for j, duration_ms in enumerate(durations_ms):
                end_ms = start_ms + duration_ms
                split_vocal = separated_vocals_audio[start_ms:end_ms]
                
                original_base_name = os.path.splitext(original_filenames[j])[0]
                split_vocal_filename = f"{original_base_name}_(vocals).wav"
                split_vocal_path = os.path.join(separator.output_dir, split_vocal_filename)
                
                split_vocal.export(split_vocal_path, format="wav")
                vocal_files.append(split_vocal_path)
                logger.info(f"    - Split and saved separated vocal to: {split_vocal_path}")
                
                start_ms = end_ms

        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}", exc_info=True)
            for file_in_chunk in chunk:
                failures.append(
                    FailedFile(filepath=str(file_in_chunk), reason=f"Failed during separation in chunk {i}: {e}")
                )

    return StepResult(successful_outputs=vocal_files, failures=failures)


def _batch_transcribe(vocal_files: List[str], temp_dir: str) -> StepResult:
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

        # Check for files that were not transcribed for some reason
        failures = []
        for vocal_file in vocal_files:
            if vocal_file not in transcriptions:
                failures.append(FailedFile(filepath=vocal_file, reason="ASR did not return a transcription for this file."))
        
        return StepResult(successful_outputs=transcriptions, failures=failures)

    except Exception as e:
        logger.error(f"Error during batch transcription: {e}", exc_info=True)
        # If the whole batch fails, all files are marked as failed.
        failures = [FailedFile(filepath=p, reason=f"Batch transcription failed: {e}") for p in vocal_files]
        return StepResult(successful_outputs={}, failures=failures)


def _batch_export(transcriptions: Dict[str, List[Dict]], output_dir: str) -> StepResult:
    """Runs final export for a dictionary of transcribed audio files."""
    logger.info("\n--- [Step 5/5] Exporting all final segments ---")
    exported_files = []
    failures = []
    for vocal_path, asr_result in transcriptions.items():
        try:
            if not asr_result:
                raise ValueError("No transcription result available.")

            vocals_audio_data = load_audio_data(vocal_path)
            base_filename = os.path.basename(vocal_path).replace("_(vocals)", "")
            final_filename_prefix = os.path.splitext(base_filename)[0]
            
            # Since we know there is only one segment in the asr_result list, we can get the fpath
            # from the return value of export_to_wav
            exported_segments = export_to_wav(
                audio_data=vocals_audio_data,
                asr_result=asr_result,
                folder_path=output_dir,
                file_name_prefix=final_filename_prefix,
            )
            
            for segment in exported_segments:
                exported_files.append(str(segment['fpath']))
            
            logger.info(f"  - Final segment exported to '{output_dir}' with prefix '{final_filename_prefix}'.")
        except Exception as e:
            logger.error(f"Failed to export {vocal_path}: {e}", exc_info=True)
            failures.append(FailedFile(filepath=vocal_path, reason=f"Export failed: {e}"))

    return StepResult(successful_outputs=exported_files, failures=failures)


def _print_summary_report(all_failures: List[FailedFile], final_success_count: int):
    """Prints a summary of the pipeline run."""
    logger.info("\n--- Pipeline Summary Report ---")
    if not all_failures:
        logger.info(f"✅ All {final_success_count} files processed successfully!")
    else:
        logger.warning(f"Pipeline completed with {len(all_failures)} failures.")
        logger.info("---")
        for failure in all_failures:
            logger.warning(f"File: {failure.filepath}")
            logger.warning(f"  Reason: {failure.reason}")
    logger.info("-----------------------------")


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
    all_failures = []

    # -- Setup --
    dirs = _setup_directories(temp_dir, output_dir)

    # -- Step 1: VAD Segmentation --
    vad_result = _run_vad_segmentation(
        input_file, dirs["vad"], min_duration, max_duration
    )
    all_failures.extend(vad_result.failures)
    if not vad_result.successful_outputs:
        logger.error("No segments produced by VAD. Halting pipeline.")
        _print_summary_report(all_failures, 0)
        return

    # -- Step 2: Batch Standardization --
    standardize_result = _batch_standardize(vad_result.successful_outputs, dirs["standardized"])
    all_failures.extend(standardize_result.failures)
    if not standardize_result.successful_outputs:
        logger.error("No segments were successfully standardized. Halting pipeline.")
        _print_summary_report(all_failures, 0)
        return

    # -- Step 3: Batch Separation --
    separator = Separator(output_dir=dirs["separated"])
    separator.load_model(separator_model_file_name)
    separate_result = _batch_separate(standardize_result.successful_outputs, separator, temp_dir)
    all_failures.extend(separate_result.failures)
    if not separate_result.successful_outputs:
        logger.error("No vocal files were successfully separated. Halting pipeline.")
        _print_summary_report(all_failures, 0)
        return

    # -- Step 4: Batch Transcription --
    transcribe_result = _batch_transcribe(separate_result.successful_outputs, temp_dir)
    all_failures.extend(transcribe_result.failures)
    if not transcribe_result.successful_outputs:
        logger.error("No segments were successfully transcribed. Halting pipeline.")
        _print_summary_report(all_failures, 0)
        return

    # -- Step 5: Batch Export --
    export_result = _batch_export(transcribe_result.successful_outputs, output_dir)
    all_failures.extend(export_result.failures)

    logger.info("\n--- Pipeline Finished ---")
    _print_summary_report(all_failures, len(export_result.successful_outputs))

# --- Main Entry Point ---

if __name__ == "__main__":
    input_audio = r"E:\tts\xtts\Knut\_raw_data\0\Knut_Im_back.mp3"
    run_pipeline(input_audio)