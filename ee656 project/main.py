import os
import time
import argparse
import numpy as np

# Import from your modules
from signal_preprocessing import preprocess_pipeline, load_dummy_data
from feature_extraction import extract_all_features

# --- Experiment Constants (per Verma et al. 2016) ---
N_CLASSES = 8
SAMPLES_PER_CLASS = 225
TOTAL_SAMPLES = N_CLASSES * SAMPLES_PER_CLASS
FS = 50000  # 50 kHz sampling rate
DURATION = 5  # 5 seconds per recording

def build_dataset(data_dir=None, use_dummy=False):
    """
    Master integration function.
    Iterates through raw acoustic recordings, preprocesses them,
    extracts features, and saves the final feature matrix and labels.
    """
    features_list = []
    labels_list = []

    print("="*60)
    print("🚀 Initiating Air Compressor Feature Extraction Pipeline")
    print("="*60)
    print(f"Total samples to process: {TOTAL_SAMPLES} ({SAMPLES_PER_CLASS} per class)")
    
    start_time = time.time()

    for class_idx in range(N_CLASSES):
        print(f"\n⚙️ Processing Class {class_idx}...")
        
        for sample_idx in range(SAMPLES_PER_CLASS):
            
            # -------------------------------------------------------------
            # STEP 1: Load Raw Data
            # -------------------------------------------------------------
            if use_dummy:
                # Generate 5 seconds of raw dummy acoustic data (250,000 samples)
                raw_signal = load_dummy_data(sampling_rate=FS, duration=DURATION)
            else:
                # IMPORTANT: Implement actual .dat file loading here once acquired
                # Example implementation:
                # file_name = f"class_{class_idx}_record_{sample_idx}.dat"
                # file_path = os.path.join(data_dir, file_name)
                # raw_signal = load_dat_file(file_path) # You will need to write this custom loader
                pass 

            # -------------------------------------------------------------
            # STEP 2: Pre-processing Pipeline
            # -------------------------------------------------------------
            # Applies 400Hz HPF, 12kHz LPF, clipping (to 1 sec), smoothing, and normalization
            clean_signal = preprocess_pipeline(raw_signal, fs=FS)

            # -------------------------------------------------------------
            # STEP 3: Feature Extraction
            # -------------------------------------------------------------
            # Extracts 286 features (Time, Freq, MWT, DWT, WPT)
            # We unpack the tuple to only grab the vector array, ignoring the names list
            feature_vector, _ = extract_all_features(clean_signal, as_vector=True)

            # -------------------------------------------------------------
            # STEP 4: Append to Dataset Matrix
            # -------------------------------------------------------------
            features_list.append(feature_vector)
            labels_list.append(class_idx)

            # --- Progress Indicator ---
            if (sample_idx + 1) % 45 == 0:
                print(f"   [{sample_idx + 1:3d}/{SAMPLES_PER_CLASS}] samples extracted...")

    # -------------------------------------------------------------
    # STEP 5: Compile and Save
    # -------------------------------------------------------------
    print("\n📦 Compiling matrices...")
    X = np.array(features_list, dtype=float)
    y = np.array(labels_list, dtype=int)

    print(f"Final Feature Matrix Shape (X): {X.shape}") # Expected: (1800, 286)
    print(f"Final Label Vector Shape   (y): {y.shape}")   # Expected: (1800,)

    np.save('features.npy', X)
    np.save('labels.npy', y)
    
    elapsed = time.time() - start_time
    print(f"\n✅ SUCCESS! Dataset saved as 'features.npy' and 'labels.npy'.")
    print(f"Total execution time: {elapsed:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build feature dataset for Acoustic Fault Diagnosis.")
    parser.add_argument('--dummy', action='store_true', help="Use synthetic dummy data instead of real .dat files")
    parser.add_argument('--dir', type=str, default='./raw_data', help="Directory containing real raw .dat files")
    args = parser.parse_args()

    if args.dummy:
        print("\n⚠ WARNING: RUNNING IN DUMMY MODE")
        print("Generating synthetic 50kHz signals instead of reading disk files.")
    else:
        print(f"\n📂 RUNNING IN REAL MODE")
        print(f"Target directory for raw data: {args.dir}")
        print("Note: Make sure your custom .dat file loader logic is uncommented in the code!")

    build_dataset(data_dir=args.dir, use_dummy=args.dummy)