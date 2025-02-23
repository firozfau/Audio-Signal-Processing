import time
import threading
from tqdm import tqdm
from ica import SpeakerSeparator
from result import ResultSaver


# Function to show a loading progress bar
def show_loader():
    """Displays a processing loader in a separate thread."""
    for _ in tqdm(range(10), desc="Processing", ncols=80):
        time.sleep(1.0)


# Start the progress loader in a separate thread
loader_thread = threading.Thread(target=show_loader)
loader_thread.start()

# === Run the Speaker Separation Process in Parallel ===
if __name__ == "__main__":
    print("\n🎙️ Please wait... Processing your audio data 🕛")

    audio_path = "source/data.wav"  # Provide the path to your audio file
    separator = SpeakerSeparator(audio_path)

    separator.load_audio()
    frequencies, times, X = separator.compute_stft()
    separator.estimate_speakers(X)
    separator.apply_ica(X)
    separator.analyze_speakers(times)

    # Save and plot results
    result_saver = ResultSaver(separator.speaker_analysis_report, separator.S_ica)
    result_saver.save_to_files()
    result_saver.plot_results()
    result_saver.plot_speaker_waveforms()

    # Stop the loading bar once processing is done
    loader_thread.join()

    print("\n✅ Processing complete! Your results are saved 🎉")
    print("\n=== ICA Speaker Segmentation Results ===")
    print(separator.speaker_analysis_report.to_string())
