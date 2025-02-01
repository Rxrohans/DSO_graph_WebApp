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
    
    # Create updated DataFrame
    updated_df = pd.DataFrame({"Time (s)": time_array, "Raw Voltage (mV)": voltage_data})
    if apply_filter:
        updated_df["Filtered Voltage (mV)"] = filtered_voltage
    
    return updated_df, time_array, voltage_data, filtered_voltage

def main():
    st.title("CSV Voltage Analyzer")
    st.write("Upload your CSV file to process and visualize voltage data.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    
    if uploaded_file:
        apply_filter = st.checkbox("Apply Low-Pass Filter")
        cutoff_frequency = 50  # Default cutoff frequency
        
        if apply_filter:
            cutoff_frequency = st.number_input("Enter Cutoff Frequency (Hz)", min_value=1, max_value=1000, value=50)
        
        with st.spinner("Processing..."):
            updated_df, time_array, voltage_data, filtered_voltage = process_csv(uploaded_file, apply_filter, cutoff_frequency)
            
            if updated_df is not None:
                # Download updated CSV
                csv_buffer = BytesIO()
                updated_df.to_csv(csv_buffer, index=False)
                st.download_button("Download Updated CSV", csv_buffer.getvalue(), "updated_data.csv", "text/csv")
                
                # Plot the graph
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(time_array, voltage_data, label="Raw Voltage (mV)", color="blue", linewidth=1, alpha=0.6)
                
                if apply_filter:
                    ax.plot(time_array, filtered_voltage, label="Filtered Voltage (mV)", color="red", linewidth=1.5)
                
                ax.set_title("Voltage vs Time", fontsize=14)
                ax.set_xlabel("Time (s)", fontsize=12)
                ax.set_ylabel("Voltage (mV)", fontsize=12)
                ax.legend()
                ax.grid(True, linestyle="--", linewidth=0.5)
                
                st.pyplot(fig)
# Footer Section (Non-fixed position to avoid layout issues)
    st.markdown("""
        ---
        <div style="text-align: center; font-size: 14px; color: gray; padding-top: 10px;">
            🚀 Built for DSO voltage graphs | for queries contact: rohans.dmvt@gmail.com
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
