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

def peak_to_peak(voltage):
    return np.max(voltage) - np.min(voltage) if voltage is not None else None

def main():
    st.title("CSV Voltage Analyzer with Signal Subtraction")
    st.write("Upload two CSV files to compare waveforms and analyze signals.")

    file1 = st.file_uploader("Upload First CSV File", type=["csv"])
    file2 = st.file_uploader("Upload Second CSV File", type=["csv"])

    apply_filter = st.checkbox("Apply Low-Pass Filter")
    cutoff_frequency = 50  # Default cutoff frequency
    
    if apply_filter:
        cutoff_frequency = st.number_input("Enter Cutoff Frequency (Hz)", min_value=1, max_value=1000, value=50)

    if file1 and file2:
        with st.spinner("Processing..."):
            time1, voltage1, filtered1, sr1 = process_csv(file1, apply_filter, cutoff_frequency)
            time2, voltage2, filtered2, sr2 = process_csv(file2, apply_filter, cutoff_frequency)
            
            if time1 is None or time2 is None:
                return
            
            # Ensure time arrays match
            min_len = min(len(time1), len(time2))
            time = time1[:min_len]
            voltage1, voltage2 = voltage1[:min_len], voltage2[:min_len]
            filtered1 = filtered1[:min_len] if filtered1 is not None else None
            filtered2 = filtered2[:min_len] if filtered2 is not None else None
            
            # Subtract waveforms
            subtracted_voltage = voltage1 - voltage2
            subtracted_filtered = filtered1 - filtered2 if filtered1 is not None and filtered2 is not None else None

            # Calculate peak-to-peak values
            ptp_voltage1 = peak_to_peak(voltage1)
            ptp_voltage2 = peak_to_peak(voltage2)
            ptp_filtered1 = peak_to_peak(filtered1)
            ptp_filtered2 = peak_to_peak(filtered2)
            ptp_subtracted = peak_to_peak(subtracted_voltage)
            ptp_subtracted_filtered = peak_to_peak(subtracted_filtered)

            # Display peak-to-peak values
            st.markdown(f"### **Peak-to-Peak Voltages**")
            st.write(f"- **File 1 (Raw):** {ptp_voltage1:.2f} mV")
            st.write(f"- **File 2 (Raw):** {ptp_voltage2:.2f} mV")
            if apply_filter:
                st.write(f"- **File 1 (Filtered):** {ptp_filtered1:.2f} mV")
                st.write(f"- **File 2 (Filtered):** {ptp_filtered2:.2f} mV")
            st.write(f"- **Subtracted Signal (Raw):** {ptp_subtracted:.2f} mV")
            if apply_filter:
                st.write(f"- **Subtracted Signal (Filtered):** {ptp_subtracted_filtered:.2f} mV")

            # Plotting graphs
            fig, axs = plt.subplots(3, 1, figsize=(10, 12))

            # Raw signals
            axs[0].plot(time, voltage1, label="File 1 - Raw", color="blue", alpha=0.7)
            axs[0].plot(time, voltage2, label="File 2 - Raw", color="green", alpha=0.7)
            axs[0].set_title("Raw Voltage Signals")
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Voltage (mV)")
            axs[0].legend()
            axs[0].grid(True)

            # Filtered signals
            if apply_filter:
                axs[1].plot(time, filtered1, label="File 1 - Filtered", color="red", alpha=0.7)
                axs[1].plot(time, filtered2, label="File 2 - Filtered", color="purple", alpha=0.7)
                axs[1].set_title("Filtered Voltage Signals")
                axs[1].set_xlabel("Time (s)")
                axs[1].set_ylabel("Voltage (mV)")
                axs[1].legend()
                axs[1].grid(True)

            # Subtracted signals
            axs[2].plot(time, subtracted_voltage, label="Subtracted (Raw)", color="black", alpha=0.7)
            if apply_filter:
                axs[2].plot(time, subtracted_filtered, label="Subtracted (Filtered)", color="orange", alpha=0.7)
            axs[2].set_title("Subtracted Signal")
            axs[2].set_xlabel("Time (s)")
            axs[2].set_ylabel("Voltage (mV)")
            axs[2].legend()
            axs[2].grid(True)

            st.pyplot(fig)

if __name__ == "__main__":
    main()
