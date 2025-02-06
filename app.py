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
    st.title("Tapping Signal Analyzer")
    st.write("Upload one or two CSV files to analyze the voltage signal.")

    file1 = st.file_uploader("Upload Noise Signal (Without Tapping) CSV (Optional)", type=["csv"])
    file2 = st.file_uploader("Upload Signal (With Tapping) CSV", type=["csv"])

    apply_filter = st.checkbox("Apply Low-Pass Filter")
    cutoff_frequency = 50  # Default cutoff frequency
    
    if apply_filter:
        cutoff_frequency = st.number_input("Enter Cutoff Frequency (Hz)", min_value=1, max_value=1000, value=50)

    if file2:  # Ensure at least the tapping signal is uploaded
        with st.spinner("Processing..."):
            time2, tapping_signal, filtered_tapping, sr2 = process_csv(file2, apply_filter, cutoff_frequency)
            
            if time2 is None:
                return
            
            if file1:  # If noise file is uploaded, perform subtraction
                time1, noise_signal, filtered_noise, sr1 = process_csv(file1, apply_filter, cutoff_frequency)

                if time1 is None:
                    return
                
                # Ensure time arrays match
                min_len = min(len(time1), len(time2))
                time = time1[:min_len]
                noise_signal, tapping_signal = noise_signal[:min_len], tapping_signal[:min_len]
                filtered_noise = filtered_noise[:min_len] if filtered_noise is not None else None
                filtered_tapping = filtered_tapping[:min_len] if filtered_tapping is not None else None
                
                # Subtract noise from signal to get actual tapping response
                actual_signal = tapping_signal - noise_signal
                actual_filtered = filtered_tapping - filtered_noise if filtered_tapping is not None and filtered_noise is not None else None

                # Calculate peak-to-peak values
                ptp_noise = peak_to_peak(noise_signal)
                ptp_tapping = peak_to_peak(tapping_signal)
                ptp_filtered_noise = peak_to_peak(filtered_noise)
                ptp_filtered_tapping = peak_to_peak(filtered_tapping)
                ptp_actual = peak_to_peak(actual_signal)
                ptp_actual_filtered = peak_to_peak(actual_filtered)

                # Display peak-to-peak values
                st.markdown(f"### **Peak-to-Peak Voltages**")
                st.write(f"- **Noise Signal (Without Tapping, Raw):** {ptp_noise:.2f} mV")
                st.write(f"- **Signal with Tapping (Raw):** {ptp_tapping:.2f} mV")
                if apply_filter:
                    st.write(f"- **Noise Signal (Filtered):** {ptp_filtered_noise:.2f} mV")
                    st.write(f"- **Signal with Tapping (Filtered):** {ptp_filtered_tapping:.2f} mV")
                st.write(f"- **Actual Signal (Tapping Only, Raw):** {ptp_actual:.2f} mV")
                if apply_filter:
                    st.write(f"- **Actual Signal (Tapping Only, Filtered):** {ptp_actual_filtered:.2f} mV")

                # Plotting graphs
                fig, axs = plt.subplots(3, 1, figsize=(10, 12))

                # Noise and Tapping Signals
                axs[0].plot(time, noise_signal, label="Noise Signal (Without Tapping)", color="blue", alpha=0.7)
                axs[0].plot(time, tapping_signal, label="Signal with Tapping", color="green", alpha=0.7)
                axs[0].set_title("Noise vs. Tapping Signal")
                axs[0].set_xlabel("Time (s)")
                axs[0].set_ylabel("Voltage (mV)")
                axs[0].legend()
                axs[0].grid(True)

                # Filtered Noise and Tapping Signals
                if apply_filter:
                    axs[1].plot(time, filtered_noise, label="Filtered Noise Signal", color="red", alpha=0.7)
                    axs[1].plot(time, filtered_tapping, label="Filtered Signal with Tapping", color="purple", alpha=0.7)
                    axs[1].set_title("Filtered Noise vs. Filtered Tapping Signal")
                    axs[1].set_xlabel("Time (s)")
                    axs[1].set_ylabel("Voltage (mV)")
                    axs[1].legend()
                    axs[1].grid(True)

                # Actual Signal (Tapping Only)
                axs[2].plot(time, actual_signal, label="Actual Signal (Tapping Only, Raw)", color="black", alpha=0.7)
                if apply_filter:
                    axs[2].plot(time, actual_filtered, label="Actual Signal (Tapping Only, Filtered)", color="orange", alpha=0.7)
                axs[2].set_title("Extracted Tapping Signal (Noise Removed)")
                axs[2].set_xlabel("Time (s)")
                axs[2].set_ylabel("Voltage (mV)")
                axs[2].legend()
                axs[2].grid(True)

                st.pyplot(fig)

            else:  # Only one file uploaded, show a single graph
                ptp_tapping = peak_to_peak(tapping_signal)
                ptp_filtered_tapping = peak_to_peak(filtered_tapping)

                st.markdown(f"### **Peak-to-Peak Voltage**")
                st.write(f"- **Signal with Tapping (Raw):** {ptp_tapping:.2f} mV")
                if apply_filter:
                    st.write(f"- **Filtered Signal with Tapping:** {ptp_filtered_tapping:.2f} mV")

                # Plot raw and filtered signal
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(time2, tapping_signal, label="Signal with Tapping (Raw)", color="green", alpha=0.7)
                if apply_filter:
                    ax.plot(time2, filtered_tapping, label="Filtered Signal with Tapping", color="red", alpha=0.7)
                ax.set_title("Tapping Signal")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Voltage (mV)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
