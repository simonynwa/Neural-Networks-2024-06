import os
import librosa
import numpy as np
import scipy.stats
import pandas as pd

def calculate_statistics(feature):
    return {
        'max': np.max(feature),
        'min': np.min(feature),
        'mean': np.mean(feature),
        'std': np.std(feature),
        'kurtosis': scipy.stats.kurtosis(feature, axis=None),
        'skew': scipy.stats.skew(feature, axis=None)
    }

def clean_feature(feature):
    # Function to clean feature data: divide by 1000 if value is greater than 100
    return np.where(feature > 100, feature / 1000, feature)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)

    # Extracting features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Clean the features
    centroid = clean_feature(centroid)
    flux = clean_feature(flux)
    rmse = clean_feature(rmse)
    zcr = clean_feature(zcr)
    contrast = clean_feature(contrast)
    bandwidth = clean_feature(bandwidth)
    flatness = clean_feature(flatness)
    rolloff = clean_feature(rolloff)
    mfccs = np.array([clean_feature(mfcc) for mfcc in mfccs])

    # Calculating statistics
    features = {}

    features['centroid'] = calculate_statistics(centroid)
    features['flux'] = calculate_statistics(flux)
    features['rmse'] = calculate_statistics(rmse)
    features['zcr'] = calculate_statistics(zcr)
    features['contrast'] = calculate_statistics(contrast)
    features['bandwidth'] = calculate_statistics(bandwidth)
    features['flatness'] = calculate_statistics(flatness)
    features['rolloff'] = calculate_statistics(rolloff)

    for i in range(20):
        features[f'mfcc_{i}'] = calculate_statistics(mfccs[i])

    features['tempo'] = tempo.item() if isinstance(tempo, np.ndarray) else tempo

    return features

def convert_to_dataframe(features, label):
    # Flatten the nested dictionaries
    flat_features = {}
    for key, stats in features.items():
        if isinstance(stats, dict):
            for stat_name, value in stats.items():
                flat_features[f'{key}_{stat_name}'] = value
        else:
            flat_features[key] = stats

    # Add the label
    flat_features['label'] = label

    # Convert to DataFrame
    df = pd.DataFrame([flat_features])
    return df

def process_all_files(root_dirs_labels):
    all_features = []

    for root_dir, label in root_dirs_labels:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(subdir, file)
                    features = extract_features(audio_path)
                    df = convert_to_dataframe(features, label)
                    all_features.append(df)

    
    final_df = pd.concat(all_features, ignore_index=True)
    return final_df

def split_train_test(df, train_size=40, test_size=10):
    train_list = []
    test_list = []

    # Split the data for each label
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        label_df = label_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data
        train_df = label_df.iloc[:train_size]
        test_df = label_df.iloc[train_size:train_size + test_size]
        
        train_list.append(train_df)
        test_list.append(test_df)

    # Concatenate all train and test dataframes
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    return train_df, test_df


root_dirs_labels = [
    ('music/blues', 0),
    ('music/classical', 1),
    ('music/pop', 2),
    ('music/jazz', 3),
    ('music/metal', 4)
]

final_df = process_all_files(root_dirs_labels)
final_df.to_csv('audio_features.csv', index=False)
train_df, test_df = split_train_test(final_df)

train_df.to_csv('train_audio_features.csv', index=False)
test_df.to_csv('test_audio_features.csv', index=False)
