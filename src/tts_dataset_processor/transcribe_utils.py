import os
import json
import csv
import librosa
import logging
from pathlib import Path
from typing import List, Dict, Union
from dataclasses import dataclass

from .asr.canary.chunked_infer import transcribe_audio_dir
from .audio_utils import convert_to_wav

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Container for transcription results with metadata."""
    audio_file: str
    text: str
    duration: float
    sample_rate: int
    basename: str


def transcribe_files(
    input_path: Union[str, List[str]], 
    asr_model_name: str = "nvidia/canary-1b-flash",
    lang: str = "en",
    output_csv: bool = True,
    output_json: bool = True,
    temp_dir: str = "tmp_transcribe"
) -> List[TranscriptionResult]:
    """
    Transcribe audio files and create metadata files.
    
    Args:
        input_path: Path to audio file, directory, or list of file paths
        asr_model_name: ASR model to use for transcription
        lang: Language code for transcription
        output_csv: Whether to create CSV metadata file
        output_json: Whether to create JSON metadata file
        temp_dir: Temporary directory for processing
        
    Returns:
        List of TranscriptionResult objects
    """
    logger.info(f"Starting transcription with model: {asr_model_name}")
    
    # Get list of audio files to process
    audio_files = _get_audio_files(input_path)
    if not audio_files:
        logger.error("No audio files found to transcribe")
        return []
    
    logger.info(f"Found {len(audio_files)} audio files to transcribe")
    
    # Create temp directory for ASR processing
    os.makedirs(temp_dir, exist_ok=True)
    
    results = []
    
    try:
        # Copy/convert files to temp directory for ASR processing
        asr_file_map = {}  # Maps temp file path to original file path
        
        for audio_file in audio_files:
            try:
                basename = os.path.basename(audio_file)
                temp_file = os.path.join(temp_dir, basename)
                
                # Convert to 16kHz mono WAV for ASR if needed
                if not audio_file.lower().endswith('.wav'):
                    temp_file = os.path.splitext(temp_file)[0] + '.wav'
                    convert_to_wav(audio_file, temp_file)
                else:
                    # Just copy if already WAV
                    import shutil
                    shutil.copy2(audio_file, temp_file)
                
                asr_file_map[temp_file] = audio_file
                
            except Exception as e:
                logger.error(f"Failed to prepare {audio_file} for ASR: {e}")
                continue
        
        if not asr_file_map:
            logger.error("No files could be prepared for transcription")
            return []
        
        # Run batch transcription on temp directory
        logger.info("Running ASR transcription...")
        asr_results = transcribe_audio_dir(
            inp_audio_dir=temp_dir,
            result_to_file=False,
            lang=lang,
            pretrained_name=asr_model_name,
        )
        
        # Process results
        for asr_result in asr_results:
            temp_file_path = asr_result.filepath
            original_file_path = asr_file_map.get(temp_file_path)
            
            if not original_file_path:
                logger.warning(f"Could not map ASR result back to original file: {temp_file_path}")
                continue
            
            try:
                # Get audio metadata
                duration = librosa.get_duration(path=original_file_path)
                sample_rate = librosa.get_samplerate(path=original_file_path)
                basename = os.path.splitext(os.path.basename(original_file_path))[0]
                
                result = TranscriptionResult(
                    audio_file=original_file_path,
                    text=asr_result.text.strip(),
                    duration=duration,
                    sample_rate=sample_rate,
                    basename=basename
                )
                
                results.append(result)
                logger.info(f"Transcribed '{basename}': '{result.text[:50]}...'")
                
            except Exception as e:
                logger.error(f"Failed to process result for {original_file_path}: {e}")
                continue
    
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    if not results:
        logger.error("No files were successfully transcribed")
        return []
    
    # Create output metadata files
    if results:
        # Determine output directory (same as first input file's directory)
        output_dir = os.path.dirname(results[0].audio_file)
        
        if output_csv:
            _create_transcription_csv(results, output_dir)
        
        if output_json:
            _create_transcription_json(results, output_dir, asr_model_name, lang)
    
    logger.info(f"Successfully transcribed {len(results)} files")
    return results


def _get_audio_files(input_path: Union[str, List[str]]) -> List[str]:
    """Get list of audio files from input path(s)."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
    audio_files = []
    
    if isinstance(input_path, list):
        # List of file paths provided
        for file_path in input_path:
            if os.path.isfile(file_path) and Path(file_path).suffix.lower() in audio_extensions:
                audio_files.append(os.path.abspath(file_path))
    
    elif os.path.isfile(input_path):
        # Single file provided
        if Path(input_path).suffix.lower() in audio_extensions:
            audio_files.append(os.path.abspath(input_path))
    
    elif os.path.isdir(input_path):
        # Directory provided - find all audio files
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if Path(file).suffix.lower() in audio_extensions:
                    audio_files.append(os.path.abspath(os.path.join(root, file)))
    
    return sorted(audio_files)


def _create_transcription_csv(results: List[TranscriptionResult], output_dir: str):
    """Create CSV file with transcription results."""
    csv_path = os.path.join(output_dir, "transcriptions.csv")
    
    try:
        with open(csv_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(["audio_file", "text", "duration", "sample_rate", "basename"])
            
            for result in sorted(results, key=lambda x: x.basename):
                relative_path = os.path.relpath(result.audio_file, output_dir)
                writer.writerow([
                    relative_path,
                    result.text,
                    f"{result.duration:.2f}",
                    result.sample_rate,
                    result.basename
                ])
        
        logger.info(f"Created transcription CSV: {csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to create CSV file: {e}")


def _create_transcription_json(results: List[TranscriptionResult], output_dir: str, asr_model: str, lang: str):
    """Create JSON file with transcription results and metadata."""
    json_path = os.path.join(output_dir, "transcriptions.json")
    
    try:
        # Build JSON structure
        json_data = {
            "metadata": {
                "total_files": len(results),
                "asr_model": asr_model,
                "language": lang,
                "total_duration": sum(r.duration for r in results),
                "average_duration": sum(r.duration for r in results) / len(results) if results else 0,
            },
            "transcriptions": []
        }
        
        for result in sorted(results, key=lambda x: x.basename):
            relative_path = os.path.relpath(result.audio_file, output_dir)
            json_data["transcriptions"].append({
                "audio_file": relative_path,
                "text": result.text,
                "duration": round(result.duration, 2),
                "sample_rate": result.sample_rate,
                "basename": result.basename
            })
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created transcription JSON: {json_path}")
        
    except Exception as e:
        logger.error(f"Failed to create JSON file: {e}")


# Convenience function for command line usage
def transcribe_cli(input_path: str, **kwargs):
    """Command line interface for transcription."""
    return transcribe_files(input_path, **kwargs)
