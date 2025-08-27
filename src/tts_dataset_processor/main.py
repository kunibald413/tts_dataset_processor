import os
import shutil
import librosa
import logging
import sys
import traceback
import json
from audio_separator.separator import Separator
from pydub import AudioSegment
from ten_vad import TenVad
from pathlib import Path
from typing import Dict, List, Iterable, Any
from dataclasses import dataclass
import numpy as np

from .audio_utils import (
    convert_to_16k_mono,
    standardization,
    export_to_wav,
    load_audio_data,
    convert_to_wav,
    convert_to_mono,
    AudioData,
)
from .vad import cut_audio_segments, detect_and_merge_speech_segments, smart_merge_small_segments
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
class ProcessedFile:
    """Represents a file with its original source tracking."""
    
    filepath: str
    original_source: str


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
        "concatenated": os.path.join(temp_dir, "4_concatenated"),
        "vad_final": os.path.join(temp_dir, "5_vad_final"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    logger.info(f"Temporary directory: {temp_dir}")
    logger.info(f"Output directory: {output_dir}")
    return dirs


def _run_vad_segmentation(
    input_file: str, vad_segments_dir: str, min_duration: float, max_duration: float,
    target_duration: float, merge_gap: float
) -> StepResult:
    """Runs VAD on the input file and saves segments to a directory."""
    logger.info("[Step 1/7] Running initial VAD segmentation...")
    try:
        sr_vad, data_vad = convert_to_16k_mono(input_file)
        if data_vad is None:
            raise ValueError("VAD pre-processing failed (could not load audio).")

        ten_vad_instance = TenVad(256, 0.5)
        initial_timestamps = detect_and_merge_speech_segments(
            ten_vad_instance, data_vad, sr_vad, ten_vad_instance.hop_size, max_duration_s=max_duration
        )
        
        speech_timestamps = smart_merge_small_segments(
            initial_timestamps, 
            min_duration=min_duration, 
            target_duration=target_duration,
            max_merge_gap=merge_gap,
            max_duration=max_duration
        )
        
        logger.info(f"  Initial VAD detected {len(initial_timestamps)} segments, smart-merged to {len(speech_timestamps)} segments")
        
        # Use the input filename stem as a prefix for VAD segments to ensure uniqueness
        file_stem = Path(input_file).stem
        cut_audio_segments(
            speech_timestamps, input_file, vad_segments_dir, min_duration, output_prefix=file_stem
        )

        segment_files = sorted([f for f in os.listdir(vad_segments_dir) if f.startswith(file_stem)])
        logger.info(f"VAD produced {len(segment_files)} segments for {os.path.basename(input_file)}.")
        
        original_source = Path(input_file).stem
        processed_files = [
            ProcessedFile(filepath=os.path.join(vad_segments_dir, f), original_source=original_source)
            for f in segment_files
        ]
        return StepResult(successful_outputs=processed_files, failures=[])

    except Exception as e:
        logger.error(f"VAD step failed entirely: {e}", exc_info=True)
        failure = FailedFile(filepath=input_file, reason=str(e))
        return StepResult(successful_outputs=[], failures=[failure])


def _concatenate_segments_by_source(
    vocal_files: List[ProcessedFile], concat_dir: str, silence_duration_ms: int = 500
) -> StepResult:
    """Concatenates separated vocal segments back by original source file with silence padding."""
    logger.info("\n--- [Step 4a/7] Concatenating separated segments by original source ---")
    concatenated_files = []
    failures = []
    
    source_groups = {}
    for processed_file in vocal_files:
        source_name = processed_file.original_source
        
        if source_name not in source_groups:
            source_groups[source_name] = []
        source_groups[source_name].append(processed_file.filepath)
    
    for source_name, segments in source_groups.items():
        try:
            logger.info(f"  - Concatenating {len(segments)} segments for {source_name}")
            
            segments.sort()
            
            combined_audio = None
            silence = AudioSegment.silent(duration=silence_duration_ms)
            
            for i, segment_file in enumerate(segments):
                segment_audio = AudioSegment.from_file(segment_file)
                
                if combined_audio is None:
                    combined_audio = segment_audio
                else:
                    combined_audio += silence + segment_audio
            
            if combined_audio is None:
                raise ValueError(f"No audio segments found for source {source_name}")
            
            concat_filename = f"{source_name}_vocals_concat.wav"
            concat_path = os.path.join(concat_dir, concat_filename)
            combined_audio.export(concat_path, format="wav")
            
            processed_concat = ProcessedFile(filepath=concat_path, original_source=source_name)
            concatenated_files.append(processed_concat)
            logger.info(f"    Created concatenated file: {concat_filename}")
            
        except Exception as e:
            logger.error(f"Failed to concatenate segments for {source_name}: {e}", exc_info=True)
            failure = FailedFile(filepath=source_name, reason=str(e))
            failures.append(failure)
    
    return StepResult(successful_outputs=concatenated_files, failures=failures)


def _run_final_vad_segmentation(
    concatenated_files: List[ProcessedFile], vad_final_dir: str, min_duration: float, max_duration: float,
    target_duration: float, merge_gap: float
) -> StepResult:
    """Runs VAD on concatenated vocal files to create training-friendly segments."""
    logger.info("\n--- [Step 4b/7] Running final VAD segmentation on concatenated vocals ---")
    all_final_segments = []
    failures = []
    
    for processed_concat in concatenated_files:
        concat_file = processed_concat.filepath
        original_source = processed_concat.original_source
        
        try:
            logger.info(f"  - Processing {os.path.basename(concat_file)} (source: {original_source})")
            sr_vad, data_vad = convert_to_16k_mono(concat_file)
            if data_vad is None:
                raise ValueError("VAD pre-processing failed (could not load audio).")

            ten_vad_instance = TenVad(256, 0.5)
            initial_timestamps = detect_and_merge_speech_segments(
                ten_vad_instance, data_vad, sr_vad, ten_vad_instance.hop_size, max_duration_s=max_duration
            )
            
            speech_timestamps = smart_merge_small_segments(
                initial_timestamps,
                min_duration=min_duration,
                target_duration=target_duration,
                max_merge_gap=merge_gap,
                max_duration=max_duration
            )
            
            logger.info(f"    Final VAD: {len(initial_timestamps)} initial → {len(speech_timestamps)} optimized segments")
            
            file_stem = Path(concat_file).stem
            cut_audio_segments(
                speech_timestamps, concat_file, vad_final_dir, min_duration, output_prefix=file_stem
            )

            segment_files = sorted([f for f in os.listdir(vad_final_dir) if f.startswith(file_stem)])
            logger.info(f"    Final VAD produced {len(segment_files)} segments.")

            for segment_file in segment_files:
                segment_path = os.path.join(vad_final_dir, segment_file)
                processed_segment = ProcessedFile(filepath=segment_path, original_source=original_source)
                all_final_segments.append(processed_segment)

        except Exception as e:
            logger.error(f"Final VAD failed for {concat_file}: {e}", exc_info=True)
            failure = FailedFile(filepath=concat_file, reason=str(e))
            failures.append(failure)

    return StepResult(successful_outputs=all_final_segments, failures=failures)


def _group_into_chunks(data: Iterable[Any], chunk_size: int) -> List[List[Any]]:
    """
    Groups an iterable into chunks of a specific size.
    If the last chunk is smaller than the chunk_size and there are at least
    two chunks, the last chunk is merged with the second to last one.
    """
    data = list(data)
    if not data:
        return []

    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Check if there are at least two chunks and the last one is smaller than the chunk size
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)

    return chunks


def _get_audio_files_from_input(input_path: str) -> List[str]:
    """Finds all audio files from a given file or directory path."""
    if not os.path.exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        return []

    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        audio_files = []
        for ext in ("*.wav", "*.mp3", "*.flac"):
            audio_files.extend(Path(input_path).rglob(ext))
        logger.info(f"Found {len(audio_files)} audio files in {input_path}.")
        return [str(p) for p in audio_files]
    
    return []


# --- Batch Processing Functions ---


def _batch_standardize(vad_segments: List[ProcessedFile], standardized_dir: str) -> StepResult:
    """Runs standardization on a list of audio files."""
    logger.info("\n--- [Step 2/7] Standardizing all segments ---")
    standardized_files = []
    failures = []
    for processed_segment in vad_segments:
        segment_path = processed_segment.filepath
        original_source = processed_segment.original_source
        try:
            segment_filename = os.path.basename(segment_path)
            standardized_data = standardization(segment_path)
            standardized_file_path = os.path.join(standardized_dir, segment_filename)
            export_to_wav(standardized_data, standardized_file_path)
            
            processed_standardized = ProcessedFile(filepath=str(standardized_file_path), original_source=original_source)
            standardized_files.append(processed_standardized)
            logger.info(f"  - Standardized: {standardized_file_path}")
        except Exception as e:
            logger.error(f"Failed to standardize {segment_path}: {e}", exc_info=True)
            failures.append(FailedFile(filepath=segment_path, reason=str(e)))

    return StepResult(successful_outputs=standardized_files, failures=failures)


def _batch_separate(standardized_files: List[ProcessedFile], separator: Separator, temp_dir: str) -> StepResult:
    """
    Runs vocal separation on a list of standardized audio files,
    grouping short files together to improve separation quality.
    """
    logger.info("\n--- [Step 3/7] Separating vocals for all segments (with chunking) ---")
    
    # Create a temporary directory for the concatenated audio chunks
    concatenated_dir = os.path.join(temp_dir, "3a_concatenated")
    os.makedirs(concatenated_dir, exist_ok=True)

    vocal_files = []
    failures = []
    
    file_paths_with_metadata = [(pf.filepath, pf.original_source) for pf in standardized_files]
    file_chunks = _group_into_chunks([fp for fp, _ in file_paths_with_metadata], 5)
    metadata_chunks = _group_into_chunks([md for _, md in file_paths_with_metadata], 5)

    for i, (chunk, metadata_chunk) in enumerate(zip(file_chunks, metadata_chunks)):
        if not chunk:
            continue

        logger.info(f"  - Processing chunk {i+1} with {len(chunk)} files...")
        
        try:
            # --- Concatenate files in the chunk, separator model can have issues with too short audios ---
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
            for j, (duration_ms, original_source) in enumerate(zip(durations_ms, metadata_chunk)):
                end_ms = start_ms + duration_ms
                split_vocal = separated_vocals_audio[start_ms:end_ms]
                
                original_base_name = os.path.splitext(original_filenames[j])[0]
                split_vocal_filename = f"{original_base_name}_(vocals).wav"
                split_vocal_path = os.path.join(separator.output_dir, split_vocal_filename)
                
                split_vocal.export(split_vocal_path, format="wav")
                
                processed_vocal = ProcessedFile(filepath=split_vocal_path, original_source=original_source)
                vocal_files.append(processed_vocal)
                logger.info(f"    - Split and saved separated vocal to: {split_vocal_path}")
                
                start_ms = end_ms

        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}", exc_info=True)
            for file_in_chunk, source in zip(chunk, metadata_chunk):
                failures.append(
                    FailedFile(filepath=str(file_in_chunk), reason=f"Failed during separation in chunk {i}: {e}")
                )

    return StepResult(successful_outputs=vocal_files, failures=failures)


def _batch_prepare_for_asr(vocal_files: List[ProcessedFile], temp_dir: str) -> Dict[str, ProcessedFile]:
    """
    Ensures all vocal files are 16kHz mono WAV for the ASR model.
    Returns a mapping of {asr_input_path: original_vocal_path}.
    """
    logger.info("\n--- [Step 5/7] Preparing files for ASR ---")
    asr_input_dir = os.path.join(temp_dir, "4_asr_input")
    if os.path.exists(asr_input_dir):
        shutil.rmtree(asr_input_dir)
    os.makedirs(asr_input_dir)

    path_map = {}
    for processed_vocal in vocal_files:
        vocal_path = processed_vocal.filepath
        try:
            audio = AudioSegment.from_file(vocal_path)
            asr_ready_path = os.path.join(asr_input_dir, os.path.basename(vocal_path))

            if audio.frame_rate != 16000 or audio.channels != 1 or not vocal_path.lower().endswith('.wav'):
                logger.info(f"  - Converting '{os.path.basename(vocal_path)}' to 16kHz mono WAV.")
                convert_to_wav(vocal_path, asr_ready_path)
            else:
                shutil.copy2(vocal_path, asr_ready_path)
            
            # Map ASR-ready file back to the original high-quality vocal file (not the 16kHz version)
            path_map[asr_ready_path] = processed_vocal
        except Exception as e:
            logger.error(f"Failed to prepare {vocal_path} for ASR: {e}", exc_info=True)
    
    return path_map


def _batch_transcribe(
    asr_path_map: Dict[str, ProcessedFile], temp_dir: str, asr_model_name: str
) -> StepResult:
    """Runs transcription on a batch of prepared vocal files."""
    logger.info(f"\n--- [Step 6/7] Transcribing all vocal segments in a single batch using {asr_model_name} ---")
    
    asr_input_files = list(asr_path_map.keys())
    if not asr_input_files:
        logger.warning("No files were successfully prepared for ASR.")
        return StepResult(successful_outputs={}, failures=[])

    try:
        asr_input_dir = os.path.dirname(asr_input_files[0])
        asr_results = transcribe_audio_dir(
            inp_audio_dir=asr_input_dir,
            result_to_file=False,
            lang="en",
            pretrained_name=asr_model_name,
        )
        
        transcriptions = {}
        for result in asr_results:
            # Map the ASR result file back to its original, high-quality vocal path
            processed_vocal = asr_path_map.get(result.filepath)
            if processed_vocal:
                original_vocal_path = processed_vocal.filepath
                duration = librosa.get_duration(path=original_vocal_path)
                transcriptions[original_vocal_path] = [{"start": 0, "end": duration, "text": result.text}]
                logger.info(f"  - Transcribed '{os.path.basename(original_vocal_path)}': '{result.text[:50]}...'")

        failures = []
        original_vocal_paths = [pf.filepath for pf in asr_path_map.values()]
        for vocal_file in original_vocal_paths:
            if vocal_file not in transcriptions:
                failures.append(FailedFile(filepath=vocal_file, reason="ASR did not return a transcription for this file."))
        
        return StepResult(successful_outputs=transcriptions, failures=failures)

    except Exception as e:
        logger.error(f"Error during batch transcription: {e}", exc_info=True)
        tb_str = traceback.format_exc()
        detailed_reason = f"Batch transcription failed: {e}\n{tb_str}"
        
        failures = []
        processed_vocals = list(asr_path_map.values())
        if processed_vocals:
            first_vocal_path = processed_vocals[0].filepath
            failures.append(FailedFile(filepath=first_vocal_path, reason=detailed_reason))
            simple_reason = f"Batch transcription failed (see traceback for {os.path.basename(first_vocal_path)})"
            for pv in processed_vocals[1:]:
                failures.append(FailedFile(filepath=pv.filepath, reason=simple_reason))
        
        return StepResult(successful_outputs={}, failures=failures)


def _batch_export(transcriptions: Dict[str, List[Dict]], wavs_output_dir: str) -> StepResult:
    """
    Exports the final audio files and their transcriptions. The final filename is
    derived from the original source file and VAD segment index.
    """
    logger.info("\n--- [Step 7/7] Exporting all final segments ---")
    exported_files = []
    failures = []
    
    sorted_items = sorted(transcriptions.items())

    for i, (vocal_path, asr_result) in enumerate(sorted_items):
        try:
            if not asr_result or "text" not in asr_result[0]:
                raise ValueError("No transcription text available.")
            
            text = asr_result[0]["text"].strip()
            if not text:
                logger.info(f"  - Skipping '{vocal_path}' (empty transcription)")
                continue

            # The vocal_path already contains the unique name (e.g., source_001_(vocals).wav).
            # We just need to clean it up for the final export.
            base_filename = os.path.basename(vocal_path).replace("_(vocals)", "")
            final_wav_path = os.path.join(wavs_output_dir, base_filename)
            
            convert_to_mono(vocal_path, final_wav_path)
            
            exported_files.append({
                "filepath": final_wav_path,
                "text": text,
                "basename": os.path.splitext(base_filename)[0],
            })

            logger.info(f"  - Exported '{vocal_path}' -> '{final_wav_path}'")
        
        except Exception as e:
            logger.error(f"Failed to export {vocal_path}: {e}", exc_info=True)
            failures.append(FailedFile(filepath=vocal_path, reason=f"Export failed: {e}"))

    return StepResult(successful_outputs=exported_files, failures=failures)


def _create_metadata_csv(export_results: List[Dict], output_dir: str, speaker_name: str):
    """Creates the final metadata.csv file for the dataset."""
    logger.info("\n--- Creating metadata.csv ---")
    csv_header = "audio_file|text|speaker_name"
    csv_path = os.path.join(output_dir, "metadata.csv")

    try:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(f"{csv_header}\n")
            for result in sorted(export_results, key=lambda x: x['basename']):
                relative_path = f"wavs/{os.path.basename(result['filepath'])}"
                text = result["text"]
                f.write(f"{relative_path}|{text}|{speaker_name}\n")
        
        logger.info(f"Successfully created metadata file at: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to create metadata.csv: {e}", exc_info=True)


def _create_metadata_json(
    export_results: List[Dict],
    output_dir: str,
    speaker_name: str,
    processing_params: Dict,
):
    """Creates a detailed metadata.json file for the dataset."""
    logger.info("\n--- Creating metadata.json ---")
    json_path = os.path.join(output_dir, "metadata.json")
    total_duration = 0
    entries = []

    for result in sorted(export_results, key=lambda x: x['basename']):
        try:
            duration = librosa.get_duration(path=result['filepath'])
            total_duration += duration
            entries.append({
                "audio_file": f"wavs/{os.path.basename(result['filepath'])}",
                "text": result["text"],
                "speaker_name": speaker_name,
                "duration": round(duration, 2),
            })
        except Exception as e:
            logger.error(f"Could not process {result['filepath']} for JSON metadata: {e}")

    metadata = {
        "processing_parameters": processing_params,
        "total_duration": round(total_duration, 2),
        "total_files": len(entries),
        "speaker_name": speaker_name,
        "entries": entries,
    }

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Successfully created metadata file at: {json_path}")
    except Exception as e:
        logger.error(f"Failed to create metadata.json: {e}", exc_info=True)


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
    input_path: str,
    temp_dir: str = "tmp",
    output_dir: str = "output",
    min_duration: float = 2.3,
    max_duration: float = 11.5,
    separator_model_file_name: str = "melband_roformer_big_beta4.ckpt",
    speaker_name: str = "coqui",
    asr_model_name: str = "nvidia/canary-1b-flash",
    initial_vad_max: float = 60.0,
    initial_target_duration: float = 30.0,
    initial_merge_gap: float = 2.0,
    final_target_ratio: float = 0.8,
    final_merge_gap: float = 0.250,
):
    """
    Runs the full audio processing pipeline in batch stages.
    1. Initial VAD (60s max) -> 2. Standardization -> 3. Separation -> 4. Final VAD (11.5s max) -> 5. ASR -> 6. Export
    """
    logger.info(f"--- Starting Pipeline for Input: {input_path} ---")
    all_failures = []

    # -- Setup --
    audio_files = _get_audio_files_from_input(input_path)
    if not audio_files:
        logger.error("No audio files found to process. Halting pipeline.")
        return
        
    dirs = _setup_directories(temp_dir, output_dir)
    wavs_output_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_output_dir, exist_ok=True)

    all_vad_segment_paths = []
    for audio_file in audio_files:
        logger.info(f"\n--- Running initial VAD on source file: {os.path.basename(audio_file)} ---")
        vad_result = _run_vad_segmentation(
            audio_file, dirs["vad"], min_duration, initial_vad_max, 
            initial_target_duration, initial_merge_gap
        )
        all_failures.extend(vad_result.failures)
        all_vad_segment_paths.extend(vad_result.successful_outputs)

    if not all_vad_segment_paths:
        logger.error("No segments produced by VAD across all files. Halting pipeline.")
        _print_summary_report(all_failures, 0)
        return

    # -- Step 2: Batch Standardization --
    standardize_result = _batch_standardize(all_vad_segment_paths, dirs["standardized"])
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

    # -- Step 4a: Concatenate segments by original source --
    concat_result = _concatenate_segments_by_source(
        separate_result.successful_outputs, dirs["concatenated"]
    )
    all_failures.extend(concat_result.failures)
    if not concat_result.successful_outputs:
        logger.error("No concatenated files were created. Halting pipeline.")
        _print_summary_report(all_failures, 0)
        return

    final_vad_result = _run_final_vad_segmentation(
        concat_result.successful_outputs, dirs["vad_final"], min_duration, max_duration,
        max_duration * final_target_ratio, final_merge_gap
    )
    all_failures.extend(final_vad_result.failures)
    if not final_vad_result.successful_outputs:
        logger.error("No segments were produced by final VAD. Halting pipeline.")
        _print_summary_report(all_failures, 0)
        return

    # -- Step 5: Prepare files for ASR --
    asr_path_map = _batch_prepare_for_asr(final_vad_result.successful_outputs, temp_dir)
    if not asr_path_map:
        logger.error("No files could be prepared for ASR. Halting pipeline.")
        _print_summary_report(all_failures, 0)
        return

    # -- Step 6: Batch Transcription --
    transcribe_result = _batch_transcribe(asr_path_map, temp_dir, asr_model_name)
    all_failures.extend(transcribe_result.failures)
    if not transcribe_result.successful_outputs:
        logger.error("No segments were successfully transcribed. Halting pipeline.")
        _print_summary_report(all_failures, 0)
        return

    # -- Step 7: Batch Export --
    export_result = _batch_export(transcribe_result.successful_outputs, wavs_output_dir)
    all_failures.extend(export_result.failures)

    # -- Final Step: Create Metadata CSV --
    if export_result.successful_outputs:
        processing_params = {
            "asr_model": asr_model_name,
            "min_duration_seconds": min_duration,
            "max_duration_seconds": max_duration,
            "separator_model": separator_model_file_name,
        }
        _create_metadata_csv(export_result.successful_outputs, output_dir, speaker_name)
        _create_metadata_json(
            export_result.successful_outputs, output_dir, speaker_name, processing_params
        )

    logger.info("\n--- Pipeline Finished ---")
    _print_summary_report(all_failures, len(export_result.successful_outputs))

# --- Main Entry Point ---

if __name__ == "__main__":
    # Example for a single file:
    # input_audio = r"E:\tts\xtts\Knut\_raw_data\0\Knut_Im_back.mp3"
    # Example for a directory:
    input_audio = r"E:\tts\xtts\Knut\_raw_data"
    run_pipeline(
        input_path=input_audio,
        speaker_name="knut",
        asr_model_name="nvidia/canary-1b",
    )