import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from io import BytesIO

def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def process_csv(file, apply_filter, cutoff_frequency):
    data = pd.read_csv(file, header=None)
    
    try:
        sampling_rate = float(data.iloc[1, 0].split(":")[1].strip("kSa/s")) * 1000
        num_points = int(data.iloc[5, 0].split(":")[1].strip())
    except Exception as e:
        st.error("Invalid file format! Make sure it follows the expected structure.")
        return None, None, None, None

    voltage_data = data.iloc[6:, 0].astype(float).values
    time_step = 1 / sampling_rate
    time_array = np.arange(0, num_points * time_step, time_step)[:len(voltage_data)]
    
    filtered_voltage = butter_lowpass_filter(voltage_data, cutoff_frequency, sampling_rate) if apply_filter else None
    
    return time_array, voltage_data, filtered_voltage, sampling_rate

def peak_to_peak(voltage):
    return np.max(voltage) - np.min(voltage) if voltage is not None else None

def annotate_peak(ax, signal):
    p2p = peak_to_peak(signal)
    txt = f"Peak-to-Peak: {p2p:.2f} mV"
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def main():
    st.title("Tapping Signal Analyzer")
    st.write("Upload one or two CSV files to analyze the voltage signal.")

    file1 = st.file_uploader("Upload Noise Signal (Without Tapping) CSV (Optional)", type=["csv"])
    file2 = st.file_uploader("Upload Signal (With Tapping) CSV", type=["csv"])

    apply_filter = st.checkbox("Apply Low-Pass Filter")
    cutoff_frequency = st.number_input("Enter Cutoff Frequency (Hz)", min_value=1, max_value=1000, value=50) if apply_filter else 50

    if file2:
        with st.spinner("Processing..."):
            time2, tapping_signal, filtered_tapping, sr2 = process_csv(file2, apply_filter, cutoff_frequency)
            if time2 is None:
                return
            
            if file1:
                time1, noise_signal, filtered_noise, sr1 = process_csv(file1, apply_filter, cutoff_frequency)
                if time1 is None:
                    return
                
                min_len = min(len(time1), len(time2))
                time, noise_signal, tapping_signal = time1[:min_len], noise_signal[:min_len], tapping_signal[:min_len]
                filtered_noise = filtered_noise[:min_len] if apply_filter and filtered_noise is not None else None
                filtered_tapping = filtered_tapping[:min_len] if apply_filter and filtered_tapping is not None else None
                
                actual_signal = tapping_signal - noise_signal
                actual_filtered = filtered_tapping - filtered_noise if apply_filter and filtered_noise is not None else None
                
                st.write("### Debug Values:")
                st.write("Max Noise:", np.max(noise_signal), "Min Noise:", np.min(noise_signal))
                st.write("Max Tapping:", np.max(tapping_signal), "Min Tapping:", np.min(tapping_signal))
                st.write("Max Extracted:", np.max(actual_signal), "Min Extracted:", np.min(actual_signal))
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.plot(time, actual_signal, label="Extracted Signal (Noise Removed)", color="black", alpha=0.7)
                ax3.set_title("Extracted Tapping Signal (Noise Removed)")
                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Voltage (mV)")
                ax3.legend()
                ax3.grid(True)
                annotate_peak(ax3, actual_signal)
                ax3.set_ylim(np.min(actual_signal) - 5, np.max(actual_signal) + 5)
                st.pyplot(fig3)

if __name__ == "__main__":
    main()

