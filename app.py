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
    except Exception as e:
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

def annotate_peak(ax, signal):
    p2p = peak_to_peak(signal)
    txt = f"Peak-to-Peak: {p2p:.2f} mV"
    # Place the annotation in the upper right corner of the plot
    ax.text(0.98, 0.95, txt, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
                
                csv_buffer = BytesIO()
                updated_df.to_csv(csv_buffer, index=False)
                st.download_button("Download Updated CSV", csv_buffer.getvalue(), "updated_data.csv", "text/csv")
                
                # Determine common y-axis limits for noise and tapping signal graphs
                combined = np.concatenate((noise_signal, tapping_signal))
                y_min, y_max = np.min(combined), np.max(combined)
                
                # Plot Noise Signal
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(time, noise_signal, label="Noise Signal (Without Tapping)", color="blue", alpha=0.7)
                ax1.set_title("Noise Signal")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Voltage (mV)")
                ax1.set_ylim(y_min, y_max)  # Set common y-axis limits
                ax1.legend()
                ax1.grid(True)
                annotate_peak(ax1, noise_signal)
                st.pyplot(fig1)
                
                # Plot Signal with Tapping
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(time, tapping_signal, label="Signal with Tapping", color="green", alpha=0.7)
                ax2.set_title("Signal with Tapping")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Voltage (mV)")
                ax2.set_ylim(y_min, y_max)  # Set common y-axis limits
                ax2.legend()
                ax2.grid(True)
                annotate_peak(ax2, tapping_signal)
                st.pyplot(fig2)
                
                # Plot Extracted (Actual) Signal
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.plot(time, actual_signal, label="Actual Signal (Tapping Only, Raw)", color="black", alpha=0.7)
                ax3.set_title("Extracted Tapping Signal (Noise Removed)")
                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Voltage (mV)")
                ax3.legend()
                ax3.grid(True)
                annotate_peak(ax3, actual_signal)
                st.pyplot(fig3)
                
                # Plot Filtered Extracted Signal if low-pass filter is applied
                if apply_filter:
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    ax4.plot(time, actual_filtered, label="Actual Signal (Tapping Only, Filtered)", color="orange", alpha=0.7)
                    ax4.set_title("Filtered Extracted Tapping Signal")
                    ax4.set_xlabel("Time (s)")
                    ax4.set_ylabel("Voltage (mV)")
                    ax4.legend()
                    ax4.grid(True)
                    annotate_peak(ax4, actual_filtered)
                    st.pyplot(fig4)
            
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
                
                # Plot Signal with Tapping
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                ax5.plot(time2, tapping_signal, label="Signal with Tapping", color="green", alpha=0.7)
                ax5.set_title("Signal with Tapping")
                ax5.set_xlabel("Time (s)")
                ax5.set_ylabel("Voltage (mV)")
                ax5.legend()
                ax5.grid(True)
                annotate_peak(ax5, tapping_signal)
                st.pyplot(fig5)
                
                # Plot Filtered Signal with Tapping if filter is applied
                if apply_filter:
                    fig6, ax6 = plt.subplots(figsize=(10, 6))
                    ax6.plot(time2, filtered_tapping, label="Filtered Signal with Tapping", color="red", alpha=0.7)
                    ax6.set_title("Filtered Signal with Tapping")
                    ax6.set_xlabel("Time (s)")
                    ax6.set_ylabel("Voltage (mV)")
                    ax6.legend()
                    ax6.grid(True)
                    annotate_peak(ax6, filtered_tapping)
                    st.pyplot(fig6)

if __name__ == "__main__":
    main()

