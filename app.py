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
    
    # Extract metadata
    try:
        sampling_rate = float(data.iloc[1, 0].split(":")[1].strip("kSa/s")) * 1000
        num_points = int(data.iloc[5, 0].split(":")[1].strip())
    except:
        st.error("Invalid file format! Make sure it follows the expected structure.")
        return None, None, None, None

    # Extract voltage data
    voltage_data = data.iloc[6:, 0].astype(float).values

    # Generate time array
    time_step = 1 / sampling_rate
    time_array = np.arange(0, num_points * time_step, time_step)[:len(voltage_data)]
    
    # Apply Butterworth low-pass filter if selected
    filtered_voltage = None
    if apply_filter:
        filtered_voltage = butter_lowpass_filter(voltage_data, cutoff_frequency, sampling_rate)
    
    return time_array, voltage_data, filtered_voltage, sampling_rate

def main():
    st.title("Tapping Signal Analyzer")
    st.write("Upload one or two CSV files to analyze the voltage signal.")
    
    # File uploads
    file1 = st.file_uploader("Upload Noise Signal (Without Tapping) CSV (Optional)", type=["csv"])
    file2 = st.file_uploader("Upload Signal (With Tapping) CSV", type=["csv"])

    # Low-pass filter option
    apply_filter = st.checkbox("Apply Low-Pass Filter")
    cutoff_frequency = 50  # Default cutoff frequency
    if apply_filter:
        cutoff_frequency = st.number_input("Enter Cutoff Frequency (Hz)", min_value=1, max_value=1000, value=50)

    if file2:  # Ensure at least the tapping signal file is uploaded
        with st.spinner("Processing..."):
            time2, tapping_signal, filtered_tapping, sr2 = process_csv(file2, apply_filter, cutoff_frequency)
            if time2 is None:
                return
            
            # When noise CSV is provided
            if file1:
                time1, noise_signal, filtered_noise, sr1 = process_csv(file1, apply_filter, cutoff_frequency)
                if time1 is None:
                    return

                # Make sure both files have the same length
                min_len = min(len(time1), len(time2))
                time = time1[:min_len]
                noise_signal = noise_signal[:min_len]
                tapping_signal = tapping_signal[:min_len]
                if apply_filter:
                    filtered_noise = filtered_noise[:min_len] if filtered_noise is not None else None
                    filtered_tapping = filtered_tapping[:min_len] if filtered_tapping is not None else None

                # Calculate the actual signal (tapping signal minus noise)
                actual_signal = tapping_signal - noise_signal
                if apply_filter and (filtered_noise is not None and filtered_tapping is not None):
                    actual_filtered = filtered_tapping - filtered_noise
                else:
                    actual_filtered = None

                # Create an updated DataFrame for download
                updated_data = {
                    "Time (s)": time,
                    "Noise Signal (mV)": noise_signal,
                    "Tapping Signal (mV)": tapping_signal,
                    "Extracted Signal (mV)": actual_signal
                }
                if apply_filter:
                    updated_data["Filtered Noise Signal (mV)"] = filtered_noise
                    updated_data["Filtered Tapping Signal (mV)"] = filtered_tapping
                    updated_data["Filtered Extracted Signal (mV)"] = actual_filtered
                updated_df = pd.DataFrame(updated_data)
                
                # Download button
                csv_buffer = BytesIO()
                updated_df.to_csv(csv_buffer, index=False)
                st.download_button("Download Updated CSV", csv_buffer.getvalue(), "updated_data.csv", "text/csv")
                
                # Determine number of subplots: 3 (without filter) or 4 (with filter)
                num_plots = 4 if apply_filter else 3
                fig, axs = plt.subplots(num_plots, 1, figsize=(10, 4*num_plots))
                
                # Plot Noise Signal
                axs[0].plot(time, noise_signal, label="Noise Signal (Without Tapping)", color="blue", alpha=0.7)
                axs[0].set_title("Noise Signal")
                axs[0].set_xlabel("Time (s)")
                axs[0].set_ylabel("Voltage (mV)")
                axs[0].legend()
                axs[0].grid(True)
                
                # Plot Signal with Tapping
                axs[1].plot(time, tapping_signal, label="Signal with Tapping", color="green", alpha=0.7)
                axs[1].set_title("Signal with Tapping")
                axs[1].set_xlabel("Time (s)")
                axs[1].set_ylabel("Voltage (mV)")
                axs[1].legend()
                axs[1].grid(True)
                
                # Plot Extracted (Actual) Signal
                axs[2].plot(time, actual_signal, label="Actual Signal (Tapping Only, Raw)", color="black", alpha=0.7)
                axs[2].set_title("Extracted Tapping Signal (Noise Removed)")
                axs[2].set_xlabel("Time (s)")
                axs[2].set_ylabel("Voltage (mV)")
                axs[2].legend()
                axs[2].grid(True)
                
                # Plot Filtered Extracted Signal if low-pass filter is applied
                if apply_filter:
                    axs[3].plot(time, actual_filtered, label="Actual Signal (Tapping Only, Filtered)", color="orange", alpha=0.7)
                    axs[3].set_title("Filtered Extracted Tapping Signal")
                    axs[3].set_xlabel("Time (s)")
                    axs[3].set_ylabel("Voltage (mV)")
                    axs[3].legend()
                    axs[3].grid(True)
                
                st.pyplot(fig)
            
            # When only signal CSV is provided
            else:
                # Create an updated DataFrame for download
                updated_data = {
                    "Time (s)": time2,
                    "Tapping Signal (mV)": tapping_signal
                }
                if apply_filter:
                    updated_data["Filtered Tapping Signal (mV)"] = filtered_tapping
                updated_df = pd.DataFrame(updated_data)
                
                csv_buffer = BytesIO()
                updated_df.to_csv(csv_buffer, index=False)
                st.download_button("Download Updated CSV", csv_buffer.getvalue(), "updated_data.csv", "text/csv")
                
                if apply_filter:
                    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
                    # Plot Signal with Tapping (Raw)
                    axs[0].plot(time2, tapping_signal, label="Signal with Tapping", color="green", alpha=0.7)
                    axs[0].set_title("Signal with Tapping")
                    axs[0].set_xlabel("Time (s)")
                    axs[0].set_ylabel("Voltage (mV)")
                    axs[0].legend()
                    axs[0].grid(True)
                    
                    # Plot Filtered Signal with Tapping
                    axs[1].plot(time2, filtered_tapping, label="Filtered Signal with Tapping", color="red", alpha=0.7)
                    axs[1].set_title("Filtered Signal with Tapping")
                    axs[1].set_xlabel("Time (s)")
                    axs[1].set_ylabel("Voltage (mV)")
                    axs[1].legend()
                    axs[1].grid(True)
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(time2, tapping_signal, label="Signal with Tapping", color="green", alpha=0.7)
                    ax.set_title("Signal with Tapping")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Voltage (mV)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
