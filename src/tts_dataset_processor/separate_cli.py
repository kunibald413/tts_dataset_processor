#!/usr/bin/env python3
"""
Standalone CLI for vocal separation only.
Handles chunking of short audio files for better separation quality.
"""

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

from audio_separator.separator import Separator
from pydub import AudioSegment


def setup_logging():
    """Configure logging for the separation CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def get_audio_files(input_path: str) -> List[str]:
    """Get all audio files from input path (file or directory)."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    
    if os.path.isfile(input_path):
        if Path(input_path).suffix.lower() in audio_extensions:
            return [input_path]
        else:
            raise ValueError(f"File {input_path} is not a supported audio format")
    
    elif os.path.isdir(input_path):
        audio_files = []
        for file in os.listdir(input_path):
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(input_path, file))
        
        if not audio_files:
            raise ValueError(f"No audio files found in directory {input_path}")
        
        return sorted(audio_files)
    
    else:
        raise ValueError(f"Input path {input_path} does not exist")


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds."""
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0


def group_into_chunks(files: List[str], min_duration_threshold: float = 10.0, chunk_size: int = 5) -> List[List[str]]:
    """
    Group audio files into chunks, prioritizing short files for concatenation.
    Files under min_duration_threshold are grouped together up to chunk_size.
    """
    short_files = []
    long_files = []
    
    # Separate short and long files
    for file_path in files:
        duration = get_audio_duration(file_path)
        if duration < min_duration_threshold:
            short_files.append(file_path)
        else:
            long_files.append(file_path)
    
    chunks = []
    
    # Group short files into chunks
    for i in range(0, len(short_files), chunk_size):
        chunk = short_files[i:i + chunk_size]
        chunks.append(chunk)
    
    # Each long file gets its own chunk
    for long_file in long_files:
        chunks.append([long_file])
    
    # Merge small last chunk with previous one if needed
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size // 2:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)
    
    return chunks


def separate_audio_chunks(
    input_files: List[str], 
    output_dir: str, 
    separator_model: str = "melband_roformer_big_beta4.ckpt",
    temp_dir: str = None
) -> List[Tuple[str, str]]:
    """
    Separate vocals from audio files using chunking for better quality.
    Returns list of (original_file, vocals_file) tuples.
    """
    logger = logging.getLogger(__name__)
    
    # Setup temporary directory
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="audio_separation_")
        cleanup_temp = True
    else:
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_temp = False
    
    try:
        # Setup directories
        concat_dir = os.path.join(temp_dir, "concatenated")
        separated_dir = os.path.join(temp_dir, "separated")
        os.makedirs(concat_dir, exist_ok=True)
        os.makedirs(separated_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize separator
        separator = Separator(output_dir=separated_dir)
        separator.load_model(separator_model)
        logger.info(f"Loaded separation model: {separator_model}")
        
        # Group files into chunks
        file_chunks = group_into_chunks(input_files)
        logger.info(f"Processing {len(input_files)} files in {len(file_chunks)} chunks")
        
        results = []
        
        for i, chunk in enumerate(file_chunks):
            if not chunk:
                continue
                
            logger.info(f"\n--- Processing chunk {i+1}/{len(file_chunks)} ({len(chunk)} files) ---")
            
            if len(chunk) == 1:
                # Single file - separate directly
                file_path = chunk[0]
                logger.info(f"  - Separating single file: {os.path.basename(file_path)}")
                
                # Get original format info
                original_audio = AudioSegment.from_file(file_path)
                orig_format = {
                    'channels': original_audio.channels,
                    'frame_rate': original_audio.frame_rate,
                    'sample_width': original_audio.sample_width
                }
                
                output_filenames = separator.separate([file_path])
                vocals_filename = next((f for f in output_filenames if "(vocals)" in f), None)
                
                if not vocals_filename:
                    logger.error(f"No vocals file found for {file_path}")
                    continue
                
                vocals_path = os.path.join(separated_dir, vocals_filename)
                if not os.path.exists(vocals_path):
                    logger.error(f"Vocals file not found at {vocals_path}")
                    continue
                
                # Load separated vocals and restore original format
                vocals_audio = AudioSegment.from_file(vocals_path)
                vocals_audio = vocals_audio.set_channels(orig_format['channels'])
                vocals_audio = vocals_audio.set_frame_rate(orig_format['frame_rate'])
                vocals_audio = vocals_audio.set_sample_width(orig_format['sample_width'])
                
                # Export with original filename and format
                original_filename = os.path.basename(file_path)
                final_vocals_path = os.path.join(output_dir, original_filename)
                
                # Export with original file extension to preserve format
                original_ext = Path(file_path).suffix.lower()
                export_format = "wav"  # Default fallback
                if original_ext in ['.mp3']:
                    export_format = "mp3"
                elif original_ext in ['.flac']:
                    export_format = "flac"
                elif original_ext in ['.m4a']:
                    export_format = "mp4"
                elif original_ext in ['.ogg']:
                    export_format = "ogg"
                
                vocals_audio.export(final_vocals_path, format=export_format)
                
                results.append((file_path, final_vocals_path))
                logger.info(f"    - Saved: {original_filename}")
            
            else:
                # Multiple files - concatenate, separate, then split
                logger.info(f"  - Concatenating {len(chunk)} files for batch separation")
                
                # Concatenate files with metadata tracking
                combined_audio = AudioSegment.empty()
                durations_ms = []
                file_stems = []
                original_formats = []  # Store original format info
                
                for file_path in chunk:
                    audio = AudioSegment.from_file(file_path)
                    combined_audio += audio
                    durations_ms.append(len(audio))
                    file_stems.append(Path(file_path).stem)
                    # Store original format info
                    original_formats.append({
                        'channels': audio.channels,
                        'frame_rate': audio.frame_rate,
                        'sample_width': audio.sample_width
                    })
                
                # Save concatenated file
                concat_filename = f"chunk_{i:03d}_concatenated.wav"
                concatenated_path = os.path.join(concat_dir, concat_filename)
                combined_audio.export(concatenated_path, format="wav")
                
                # Separate the concatenated file
                output_filenames = separator.separate([concatenated_path])
                vocals_filename = next((f for f in output_filenames if "(vocals)" in f), None)
                
                if not vocals_filename:
                    logger.error(f"No vocals file found for concatenated chunk {i}")
                    continue
                
                vocals_path = os.path.join(separated_dir, vocals_filename)
                if not os.path.exists(vocals_path):
                    logger.error(f"Vocals file not found at {vocals_path}")
                    continue
                
                # Split the separated vocals back to individual files
                separated_vocals_audio = AudioSegment.from_file(vocals_path)
                start_ms = 0
                
                for j, (duration_ms, file_stem, original_file, orig_format) in enumerate(zip(durations_ms, file_stems, chunk, original_formats)):
                    end_ms = start_ms + duration_ms
                    split_vocal = separated_vocals_audio[start_ms:end_ms]
                    
                    # Restore original format (channels, sample rate, bit depth)
                    split_vocal = split_vocal.set_channels(orig_format['channels'])
                    split_vocal = split_vocal.set_frame_rate(orig_format['frame_rate'])
                    split_vocal = split_vocal.set_sample_width(orig_format['sample_width'])
                    
                    # Use exact original filename and preserve original format
                    original_filename = os.path.basename(original_file)
                    final_vocals_path = os.path.join(output_dir, original_filename)
                    
                    # Export with original file extension to preserve format
                    original_ext = Path(original_file).suffix.lower()
                    export_format = "wav"  # Default fallback
                    if original_ext in ['.mp3']:
                        export_format = "mp3"
                    elif original_ext in ['.flac']:
                        export_format = "flac"
                    elif original_ext in ['.m4a']:
                        export_format = "mp4"
                    elif original_ext in ['.ogg']:
                        export_format = "ogg"
                    
                    split_vocal.export(final_vocals_path, format=export_format)
                    
                    results.append((original_file, final_vocals_path))
                    logger.info(f"    - Split and saved: {original_filename}")
                    
                    start_ms = end_ms
        
        logger.info(f"\n--- Separation Complete ---")
        logger.info(f"Successfully separated {len(results)} files to: {output_dir}")
        
        return results
    
    finally:
        if cleanup_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Main CLI entry point for vocal separation."""
    parser = argparse.ArgumentParser(
        description="Separate vocals from audio files with intelligent chunking for better quality"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="Input audio file or directory containing audio files"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Output directory for separated vocal files"
    )
    parser.add_argument(
        "-m", "--model",
        default="melband_roformer_big_beta4.ckpt",
        help="Separation model to use (default: melband_roformer_big_beta4.ckpt)"
    )
    parser.add_argument(
        "-t", "--temp-dir",
        help="Temporary directory for processing (optional, uses system temp if not specified)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="Minimum duration threshold for chunking (default: 10.0 seconds)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Maximum files per chunk (default: 5)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Get input files
        audio_files = get_audio_files(args.input)
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Run separation
        results = separate_audio_chunks(
            input_files=audio_files,
            output_dir=args.output,
            separator_model=args.model,
            temp_dir=args.temp_dir
        )
        
        # Print summary
        if results:
            logger.info(f"\nSeparation completed successfully!")
            logger.info(f"Processed {len(results)} files")
            logger.info(f"Output directory: {args.output}")
        else:
            logger.error("No files were successfully separated")
            return 1
            
    except Exception as e:
        logger.error(f"Separation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
