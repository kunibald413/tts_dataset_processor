from ten_vad import TenVad
from audio_separator.separator import Separator

from .vad import convert_to_16k_mono, cut_audio_segments, detect_and_merge_speech_segments

# Initialize the Separator class (with optional configuration properties, below)
separator = Separator()

# Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
separator.load_model()

# Perform the separation on specific audio files without reloading the model
# Create a dummy audio file for testing
with open("audio1.wav", "w") as f:
    f.write("dummy audio data")
output_files = separator.separate('audio1.wav')

print(f"Separation complete! Output file(s): {' '.join(output_files)}")

if __name__ == "__main__":
    pass