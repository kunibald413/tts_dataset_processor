import os
import shutil
import librosa
from audio_separator.separator import Separator
from ten_vad import TenVad
from pathlib import Path

from .audio_utils import (
    convert_to_16k_mono,
    standardization,
    export_to_wav,
    AudioData,
)
from .vad import cut_audio_segments, detect_and_merge_speech_segments


def transcribe_audio(audio_path: str):
    """
    Dummy function for audio transcription.
    In a real implementation, this would call an ASR model.
    """
    print(f"  - (DUMMY) Transcribing: {audio_path}")
    # Calculate duration to make the dummy data realistic
    try:
        duration = librosa.get_duration(path=audio_path)
        return [
            {
                "start": 0,
                "end": duration,
                "text": "This is a dummy transcription of the audio segment.",
            }
        ]
    except Exception as e:
        print(f"    - Could not get duration of {audio_path}: {e}")
        return []


def run_pipeline(
    input_file: str,
    temp_dir: str = "tmp",
    output_dir: str = "output",
    min_duration: float = 2.5,
    max_duration: float = 11.5,
):
    """
    Runs the full audio processing pipeline.
    1. VAD Segmentation -> 2. Standardization -> 3. Separation -> 4. ASR -> 5. Export
    """
    print(f"--- Starting Pipeline for: {input_file} ---")

    # -- Setup --
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    vad_segments_dir = os.path.join(temp_dir, "1_vad_segments")
    standardized_dir = os.path.join(temp_dir, "2_standardized")
    separated_dir = os.path.join(temp_dir, "3_separated")
    os.makedirs(vad_segments_dir, exist_ok=True)
    os.makedirs(standardized_dir, exist_ok=True)
    os.makedirs(separated_dir, exist_ok=True)

    # -- Step 1: VAD Segmentation --
    print("\n[Step 1/5] Running VAD segmentation...")
    sr_vad, data_vad = convert_to_16k_mono(input_file)
    if data_vad is None:
        print("  - VAD failed, cannot continue.")
        return

    ten_vad_instance = TenVad(256, 0.5)
    speech_timestamps = detect_and_merge_speech_segments(
        ten_vad_instance, data_vad, sr_vad, 256, max_duration_s=max_duration
    )
    cut_audio_segments(speech_timestamps, input_file, vad_segments_dir, min_duration)
    print(f"  - VAD produced {len(os.listdir(vad_segments_dir))} segments.")

    # -- Initialize Separator --
    separator = Separator(output_dir=separated_dir)
    separator.load_model()

    # -- Processing Loop --
    vad_segment_files = sorted(os.listdir(vad_segments_dir))
    for i, segment_filename in enumerate(vad_segment_files):
        print(f"\n--- Processing segment {i+1}/{len(vad_segment_files)}: {segment_filename} ---")
        segment_path = os.path.join(vad_segments_dir, segment_filename)

        # -- Step 2: Standardize --
        print("[Step 2/5] Standardizing segment...")
        standardized_data = standardization(segment_path)
        # Standardization returns float32, separator needs a file, so we save it.
        standardized_path = os.path.join(standardized_dir, segment_filename)
        export_to_wav(
            audio_data=standardized_data,
            asr_result=[{"start": 0, "end": standardized_data.waveform.shape[0] / standardized_data.sample_rate}],
            folder_path=standardized_dir,
            file_name_prefix=os.path.splitext(segment_filename)[0],
        )
        # export_to_wav creates files like `prefix_00000.wav`, so we need to find the new path
        standardized_file_path = next(Path(standardized_dir).glob(f"{os.path.splitext(segment_filename)[0]}*.wav"))
        print(f"  - Standardized segment saved to: {standardized_file_path}")

        # -- Step 3: Separate Vocals/Instruments --
        print("[Step 3/5] Separating vocals from instruments...")
        try:
            output_paths = separator.separate(standardized_file_path)
            # Find the vocals file. The library adds "(Vocals)" to the filename.
            vocals_path = next((p for p in output_paths if "(Vocals)" in p), None)
            if not vocals_path:
                print("  - Could not find vocals file in separator output. Skipping segment.")
                continue
            print(f"  - Vocals separated to: {vocals_path}")
        except Exception as e:
            print(f"  - Error during separation: {e}. Skipping segment.")
            continue

        # -- Step 4: Transcribe (ASR) --
        print("[Step 4/5] Transcribing vocals...")
        asr_result = transcribe_audio(vocals_path)
        if not asr_result:
            print("  - Transcription failed. Skipping segment.")
            continue

        # -- Step 5: Export Final Audio --
        print("[Step 5/5] Exporting final audio segment...")
        # We need the audio data from the vocals file to export it
        vocals_audio_data = standardization(vocals_path)
        final_filename_prefix = os.path.splitext(segment_filename)[0]
        export_to_wav(
            audio_data=vocals_audio_data,
            asr_result=asr_result,
            folder_path=output_dir,
            file_name_prefix=final_filename_prefix,
        )
        print(f"  - Final segment exported to '{output_dir}' with prefix '{final_filename_prefix}'.")

    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    input_audio = r"E:\tts\xtts\Knut\_raw_data\0\Knut_Im_back.mp3"
    run_pipeline(input_audio)