import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import librosa

# Assume logger and cfg are configured elsewhere in your application
# For example:
# from your_logging_module import logger
# from your_config_module import cfg

# A counter for unnamed audio segments
audio_count = 0


@dataclass
class AudioData:
    """A container for processed audio data."""

    waveform: np.ndarray
    name: str
    sample_rate: int


def standardization(audio: Union[str, AudioSegment]) -> AudioData:
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        AudioData: An object containing the preprocessed audio waveform, file name, and sample rate.

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")
    # logger.debug("Entering the preprocessing of audio")


    # Convert the audio file to WAV format
    # audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    audio = audio.set_channels(1)  # Set to mono

    # logger.debug("Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    # logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude  # Normalize

    # logger.debug(f"waveform shape: {waveform.shape}")
    # logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return AudioData(
        waveform=waveform,
        name=name,
        sample_rate=audio.frame_rate,
    )


def load_audio_data(file_path: str) -> AudioData:
    """
    Loads an audio file and returns an AudioData object without standardization.
    The waveform is normalized to the [-1.0, 1.0] range.
    """
    audio = AudioSegment.from_file(file_path)
    # Convert to a numpy array of int16 samples
    waveform_int = np.array(audio.get_array_of_samples())
    
    # Normalize to float32 in the range [-1.0, 1.0]
    waveform_float = waveform_int.astype(np.float32) / 32768.0

    return AudioData(
        waveform=waveform_float,
        name=os.path.basename(file_path),
        sample_rate=audio.frame_rate,
    )


def convert_to_wav(input_path: str, output_path: str):
    """
    Converts an audio file to 16kHz mono WAV format for ASR.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")


def export_to_wav(audio_data: AudioData, output_path: str):
    """Saves an AudioData object to a WAV file."""
    folder_path = os.path.dirname(output_path)
    os.makedirs(folder_path, exist_ok=True)

    # The waveform is now expected to be int16
    waveform = audio_data.waveform

    # Ensure audio is mono
    if waveform.ndim > 1 and waveform.shape[1] > 1:
        waveform = librosa.to_mono(waveform)

    # Write the int16 data directly.
    write_wav(output_path, audio_data.sample_rate, waveform)

    return output_path


def convert_to_16k_mono(input_file: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Converts an audio file (WAV, MP3, etc.) to 16kHz mono.

    Args:
        input_file (str): Path to the input audio file.

    Returns:
        tuple: A tuple containing the 16kHz mono audio data (np.ndarray) and the sample rate (16000).
    """
    try:
        audio = AudioSegment.from_file(input_file)
        print(f"Original audio: {audio.frame_rate}Hz, {'stereo' if audio.channels > 1 else 'mono'}")

        # Convert to mono if it's stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
            print("Converted to mono.")

        # Resample to 16kHz if necessary
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
            print("Resampled to 16kHz.")

        # Get audio data as a numpy array
        data = np.array(audio.get_array_of_samples()).astype(np.int16)
        sr = 16000

        return sr, data

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        print("Please ensure that ffmpeg is installed and in your system's PATH.")
        return None, None



def write_wav(path, sr, x):
    """Write numpy array to WAV file."""
    sf.write(path, x, sr)