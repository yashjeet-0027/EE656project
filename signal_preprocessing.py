import numpy as np
from scipy.signal import butter, filtfilt

def load_dummy_data(sampling_rate=50000, duration=5):
    """
    Simulates loading a raw acoustic recording for testing purposes.
    Generates 5 seconds of noisy data at 50 kHz.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Simulate a signal with low freq (fan noise), main signal, and high freq noise
    raw_signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 3000 * t) + np.random.normal(0, 0.5, len(t))
    return raw_signal

def filter_signal(signal, fs):
    """
    Step 1: Filtering.
    Applies a 400 Hz high-pass filter and a 12 kHz, 18th-order low-pass filter.
    """
    nyq = 0.5 * fs

    # High-pass filter (Cutoff: 400 Hz)
    hp_cutoff = 400 / nyq
    b_hp, a_hp = butter(N=4, Wn=hp_cutoff, btype='highpass') # Standard order used for fan filter
    hp_filtered = filtfilt(b_hp, a_hp, signal)

    # Low-pass filter (Cutoff: 12 kHz, Order: 18)
    lp_cutoff = 12000 / nyq
    b_lp, a_lp = butter(N=18, Wn=lp_cutoff, btype='lowpass')
    filtered_signal = filtfilt(b_lp, a_lp, hp_filtered)

    return filtered_signal

def clip_signal(signal, fs):
    """
    Step 2: Clipping.
    Divides the 5s signal into 9 overlapping 1s windows and selects the one with min standard deviation.
    """
    window_length = fs * 1  # 1 second = 50,000 samples
    step_size = int(fs * 0.5)  # 50% overlap = 25,000 samples

    num_windows = 9
    windows = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        windows.append(signal[start_idx:end_idx])

    # Find the window with the minimum standard deviation
    std_devs = [np.std(w) for w in windows]
    best_window_idx = np.argmin(std_devs)

    return windows[best_window_idx]

def smooth_signal(signal):
    """
    Step 3: Smoothing.
    Applies a moving average filter with q=2 (window size 5).
    """
    q = 2
    window_size = 2 * q + 1
    # Create an array of ones divided by the window size for the moving average
    kernel = np.ones(window_size) / window_size

    # 'same' mode keeps the output array the same size as the input
    smoothed_signal = np.convolve(signal, kernel, mode='same')
    return smoothed_signal

def normalize_signal(signal):
    """
    Step 4: Normalization.
    Excludes top/bottom 0.025% of points to find X_min and X_max, then scales to [-1, 1].
    """
    # Find outlier-resistant max and min using percentiles
    X_min = np.percentile(signal, 0.025)
    X_max = np.percentile(signal, 99.975)

    # Clip the signal to these bounds just in case there are extreme spikes
    clipped_signal = np.clip(signal, X_min, X_max)

    # Scale to [-1, 1] range: a = -1, b = 1
    a, b = -1, 1
    normalized_signal = a + ((clipped_signal - X_min) * (b - a)) / (X_max - X_min)

    return normalized_signal

def preprocess_pipeline(raw_signal, fs=50000):
    """
    Master function tying all pre-processing steps together.
    Your teammates will call this function.
    """
    # 1. Filter
    filtered = filter_signal(raw_signal, fs)

    # 2. Clip
    clipped = clip_signal(filtered, fs)

    # 3. Smooth
    smoothed = smooth_signal(clipped)

    # 4. Normalize
    final_signal = normalize_signal(smoothed)

    return final_signal

if __name__ == "__main__":
    print("Testing the preprocessing pipeline...")
    fs = 50000
    raw_data = load_dummy_data(sampling_rate=fs, duration=5)
    processed_data = preprocess_pipeline(raw_data, fs=fs)
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed data range: [{processed_data.min()}, {processed_data.max()}]")
    print("Pipeline executed successfully!")
