import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ResultSaver:
    def __init__(self, analysis_report, S_ica):
        self.analysis_report = analysis_report
        self.S_ica = S_ica

    def save_to_files(self, filename="ica_speaker_analysis"):
        """Saves the speaker analysis report as CSV and Excel."""
        csv_filename = f"{filename}.csv"
        excel_filename = f"{filename}.xlsx"

        self.analysis_report.to_csv(csv_filename, index=False)
        self.analysis_report.to_excel(excel_filename, index=False)

        print(f"\n📁 Results saved to {csv_filename} and {excel_filename}")

    def plot_results(self):
        """Plots and saves the speaker contribution pie chart."""
        plt.figure(figsize=(8, 6))
        plt.pie(
            self.analysis_report["Energy Contribution (%)"],
            labels=self.analysis_report["Speaker"],
            autopct="%1.1f%%",
            startangle=140,
        )
        plt.title("Speaker Contribution (%)")
        plt.savefig("speaker_contribution.png")
        print("📸 Speaker contribution pie chart saved as 'speaker_contribution.png'")
        plt.show()

    def plot_speaker_waveforms(self):
        """Plots and saves the separated speaker waveforms."""
        plt.figure(figsize=(12, 6))
        for i in range(self.S_ica.shape[1]):
            plt.plot(
                self.S_ica[:, i],
                label=f"Speaker {i+1} ({self.analysis_report['Energy Contribution (%)'][i]:.2f}%)",
            )

        plt.title("Separated Audio Components - Detected Speakers")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig("separated_speakers.png")  # Save as image
        print("📸 Separated speakers image saved as 'separated_speakers.png'")
        plt.show()
