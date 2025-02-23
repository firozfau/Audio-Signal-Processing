import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import stft
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd


class SpeakerSeparator:
    def __init__(self, audio_path, downsample_factor=4, min_speakers=2, max_speakers=6):
        self.audio_path = audio_path
        self.downsample_factor = downsample_factor
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.sample_rate = None
        self.audio_data = None
        self.optimal_speakers = None
        self.S_ica = None
        self.speaker_analysis_report = None

    def load_audio(self):
        """Loads and preprocesses the audio file."""
        self.sample_rate, self.audio_data = wav.read(self.audio_path)

        # Convert to mono if needed
        if len(self.audio_data.shape) > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)

        # Normalize and downsample
        self.audio_data = self.audio_data / np.max(np.abs(self.audio_data))
        self.audio_data = self.audio_data[:: self.downsample_factor]
        self.sample_rate = self.sample_rate // self.downsample_factor

    def compute_stft(self):
        """Computes the Short-Time Fourier Transform (STFT) of the audio."""
        frequencies, times, Zxx = stft(self.audio_data, fs=self.sample_rate)
        return frequencies, times, np.abs(Zxx.T)

    def estimate_speakers(self, X):
        """Estimates the optimal number of speakers using Gaussian Mixture Model (GMM)."""
        bic_scores = []
        possible_speakers = range(self.min_speakers, self.max_speakers + 1)

        for n in possible_speakers:
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))

        # Select the best number of speakers based on lowest BIC score
        self.optimal_speakers = possible_speakers[np.argmin(bic_scores)]
        print(f"\n🔍 Estimated Number of Speakers: {self.optimal_speakers}")

    def apply_ica(self, X):
        """Applies ICA for source separation."""
        ica = FastICA(n_components=self.optimal_speakers, random_state=42)
        self.S_ica = ica.fit_transform(X)

    def analyze_speakers(self, times):
        """Analyzes speaker contributions, start times, and end times."""
        # Apply clustering to identify speaker patterns
        kmeans = KMeans(n_clusters=self.optimal_speakers, random_state=42)
        clusters = kmeans.fit_predict(self.S_ica)

        # Compute energy contribution per speaker
        component_energies = np.sum(np.abs(self.S_ica), axis=0)
        total_energy = np.sum(component_energies)
        percentage_contribution = (component_energies / total_energy) * 100

        # Identify time range for each speaker
        speaker_times = {f"Speaker {i+1}": [] for i in range(self.optimal_speakers)}

        for t, cluster_label in enumerate(clusters):
            speaker_times[f"Speaker {cluster_label+1}"].append(times[t])

        speaker_time_ranges = {
            speaker: (
                (round(min(time_points), 2), round(max(time_points), 2))
                if time_points
                else ("Not detected", "Not detected")
            )
            for speaker, time_points in speaker_times.items()
        }

        # Create a DataFrame for the speaker analysis report
        self.speaker_analysis_report = pd.DataFrame(
            {
                "Speaker": [f"Speaker {i+1}" for i in range(self.optimal_speakers)],
                "Energy Contribution (%)": percentage_contribution,
                "Start Time (s)": [
                    speaker_time_ranges[f"Speaker {i+1}"][0]
                    for i in range(self.optimal_speakers)
                ],
                "End Time (s)": [
                    speaker_time_ranges[f"Speaker {i+1}"][1]
                    for i in range(self.optimal_speakers)
                ],
            }
        )
