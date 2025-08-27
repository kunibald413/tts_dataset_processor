
import os
from typing import List, Dict, Tuple, Optional

from ten_vad import TenVad
import scipy.io.wavfile as Wavfile
import numpy as np
from pydub import AudioSegment

from .audio_utils import convert_to_16k_mono


def cut_audio_segments(
    segments: List[Dict[str, float]],
    input_file: str,
    output_folder: str,
    min_duration: float = 2.0,
    output_prefix: str = "segment",
):
    """
    Cuts the input audio file into segments and saves them.
    This function loads the original audio file to preserve its sample rate for cutting.

    Args:
        segments (list): A list of dictionaries, where each dictionary
                         has 'start' and 'end' keys representing the
                         segment's time in seconds.
        input_file (str): Path to the original input audio file (WAV, MP3, etc.).
        output_folder (str): Path to the folder to save the segmented audio files.
        min_duration (float): The minimum duration for a segment to be saved. Defaults to 2.0.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    try:
        # Use pydub to load the original audio, which handles various formats
        original_audio = AudioSegment.from_file(input_file)
        original_sr = original_audio.frame_rate

        # Convert to mono for consistent processing
        if original_audio.channels > 1:
            original_audio = original_audio.set_channels(1)

        original_data = np.array(original_audio.get_array_of_samples()).astype(np.int16)

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the audio file: {e}")
        return

    print(f"\nSaving {len(segments)} audio segments...")
    for i, segment in enumerate(segments):
        start_time = segment['start']
        end_time = segment['end']

        # Calculate start and end frame indices using the original sample rate
        start_frame = int(start_time * original_sr)
        end_frame = int(end_time * original_sr)

        segment_duration = end_time - start_time
        if segment_duration < min_duration:
            print(f"segment too short: {segment_duration:.3f}s, skip!")
            continue

        # Ensure indices are within bounds
        start_frame = max(0, start_frame)
        end_frame = min(len(original_data), end_frame)

        # Extract the audio data for the current segment
        segment_data = original_data[start_frame:end_frame]

        # Create a nice padded filename, e.g., 'prefix_001.wav'
        padded_index = str(i).zfill(4)
        output_filename = f"{output_prefix}_{padded_index}.wav"
        output_path = os.path.join(output_folder, output_filename)

        # Save the segment as a new WAV file using the original sample rate
        try:
            Wavfile.write(output_path, original_sr, segment_data)
            print(f"  - Saved '{output_filename}' (duration: {end_time - start_time:.2f}s)")
        except Exception as e:
            print(f"  - Error saving segment {i}: {e}")
            continue

    print("All segments have been saved.")


def detect_and_merge_speech_segments(
    ten_vad_instance, 
    data, 
    sr, 
    hop_size, 
    merge_gap_s=0.300, 
    max_duration_s=None, 
    verbose=False
):
    """
    Detects speech segments from audio data using TenVad, merges segments with short gaps,
    and splits segments that exceed a maximum duration.

    Args:
        ten_vad_instance (TenVad): An instance of the TenVad class.
        data (np.ndarray): The input audio data.
        sr (int): The sample rate of the audio.
        hop_size (int): The hop size for processing frames.
        merge_gap_s (float): The maximum gap in seconds between segments to be merged.
        max_duration_s (float, optional): The maximum duration for a single segment.
                                           Segments longer than this will be split. Defaults to None.
        verbose (bool): If True, prints detailed information about detected and merged segments.

    Returns:
        list: A list of dictionaries, where each dictionary represents a final speech segment
              with 'start' and 'end' times in seconds.
    """
    num_frames = data.shape[0] // hop_size
    segments = []
    is_speech_active = False
    speech_active_start = 0.0

    print("Processing audio frames for speech detection...")
    for i in range(num_frames):
        audio_time_seconds = (i * hop_size) / sr
        audio_data = data[i * hop_size: (i + 1) * hop_size]
        _out_probability, out_flag = ten_vad_instance.process(audio_data)

        if out_flag == 1 and not is_speech_active:
            speech_active_start = audio_time_seconds
            is_speech_active = True
        elif out_flag == 0 and is_speech_active:
            is_speech_active = False
            segment = {"start": speech_active_start, "end": audio_time_seconds}
            segments.append(segment)

    # Check if speech was active at the very end of the audio
    if is_speech_active:
        segment = {
            "start": speech_active_start,
            "end": (num_frames * hop_size) / sr
        }
        segments.append(segment)

    # Smartly merge segments, prioritizing natural silences as split points.
    merged_segments = []
    if segments:
        current_merged = segments[0].copy()
        for i in range(1, len(segments)):
            next_segment = segments[i]
            gap = next_segment["start"] - current_merged["end"]

            should_merge = gap < merge_gap_s
            if should_merge and max_duration_s and (next_segment["end"] - current_merged["start"] > max_duration_s):
                should_merge = False

            if should_merge:
                current_merged["end"] = next_segment["end"]
            else:
                merged_segments.append(current_merged)
                current_merged = next_segment.copy()
        merged_segments.append(current_merged)

    # For verbose logging, keep a reference before the final splitting pass
    merged_segments_for_log = merged_segments

    # Split any segments that are still too long (can happen with long initial detections)
    if max_duration_s:
        final_segments = []
        for segment in merged_segments:
            duration = segment["end"] - segment["start"]
            if duration > max_duration_s:
                current_start = segment["start"]
                while current_start < segment["end"]:
                    current_end = min(current_start + max_duration_s, segment["end"])
                    final_segments.append({"start": current_start, "end": current_end})
                    current_start = current_end
            else:
                final_segments.append(segment)
        processed_segments = final_segments
    else:
        processed_segments = merged_segments

    if verbose:
        print(f"\nInitial detected segments: {len(segments)}")
        print(f"Merged segments (gaps < {merge_gap_s * 1000:.0f}ms): {len(merged_segments_for_log)}")
        if max_duration_s:
            print(f"Final segments (max duration {max_duration_s}s): {len(processed_segments)}")
        print(processed_segments)

    return processed_segments


def _run_example():
    input_file = r"E:\tts\xtts\Knut\_raw_data\0\Knut_Im_back.mp3"
    sr, data = convert_to_16k_mono(input_file)

    assert data is not None, f"data is missing {data}"
    if data is not None:
        hop_size = 256  # 16 ms per frame at 16kHz
        threshold = 0.5
        ten_vad_instance = TenVad(hop_size, threshold)

        merged_segments = detect_and_merge_speech_segments(
            ten_vad_instance, data, sr, hop_size, merge_gap_s=0.150, max_duration_s=11.5, verbose=True
        )

        cut_audio_segments(merged_segments, input_file, "tmp/audio_segments", min_duration=2.5)


if __name__ == "__main__":
    _run_example()