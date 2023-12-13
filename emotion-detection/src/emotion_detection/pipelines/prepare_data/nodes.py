from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def select_features(data: pd.DataFrame) -> list[pd.DataFrame]:
    other_data = data.drop(["dispEmo", "clipName"], axis=1)
    video_names = data["clipName"]
    target = data["dispEmo"]
    emotions_map = {
        "N": "neutral",
        "A": "angry",
        "H": "happy",
        "S": "sad",
        "D": "disgust",
        "F": "fear",
    }
    target = target.apply(lambda x: emotions_map[x])
    return video_names, target, other_data


def extract_features(audio: np.ndarray, sample_rate: float) -> list[np.ndarray]:
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate)
    mfccs = np.mean(mfccs.T, axis=0)

    stft = np.abs(librosa.stft(audio))
    stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    stft = np.mean(stft.T, axis=0)

    lpc_coef = librosa.lpc(audio, order=12)[1:]

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel = np.mean(mel.T, axis=0)

    return mfccs, stft, lpc_coef, mel


def label_to_encoded(label_array: np.ndarray, label_dict: dict) -> np.ndarray:
    result = np.zeros((len(label_array), len(label_dict)), dtype=int)
    for index, label in enumerate(label_array["dispEmo"]):
        one_hot = np.zeros(len(label_dict), dtype=int)
        one_hot[label_dict[label]] = 1
        result[index] = one_hot
    return result


def process_audio(name: str) -> tuple:
    audio_path = Path(f"../CREMA-D/AudioWAV/{name}.wav")
    audio_array, sample_rate = librosa.load(str(audio_path), duration=2)
    return extract_features(audio_array, sample_rate)


def read_data(df: pd.DataFrame, size: int = 0) -> tuple[np.ndarray]:
    data_size = df.shape[0] if not size else size
    mfccs_array = np.zeros((data_size, 20))
    stft_array = np.zeros((data_size, 12))
    lpc_array = np.zeros((data_size, 12))
    mel_array = np.zeros((data_size, 128))
    df_in = df.iloc[:data_size].to_numpy()
    with Pool() as pool:
        results = pool.imap(process_audio, df_in)
        for index, (mfccs, stft, lpc_coef, mel) in enumerate(
            tqdm(results, total=data_size)
        ):
            mfccs_array[index] = mfccs
            stft_array[index] = stft
            lpc_array[index] = lpc_coef
            mel_array[index] = mel
    return mfccs_array, stft_array, lpc_array, mel_array


def prepare_data(df: pd.DataFrame, recreate_data: bool) -> tuple[np.ndarray]:
    if not recreate_data:
        mfcc = np.loadtxt("../mfcc_features.csv", delimiter="|")
        stft = np.loadtxt("../stft_features.csv", delimiter="|")
        lpc = np.loadtxt("../lpc_features.csv", delimiter="|")
        mel = np.loadtxt("../mel_features.csv", delimiter="|")

    else:
        print("Recreating data...")
        names = df["clipName"]
        mfcc, stft, lpc, mel = read_data(names)

    mfcc = pd.DataFrame(mfcc, columns=[f"mfcc_{i}" for i in range(mfcc.shape[1])])
    stft = pd.DataFrame(stft, columns=[f"stft_{i}" for i in range(stft.shape[1])])
    lpc = pd.DataFrame(lpc, columns=[f"lpc_{i}" for i in range(lpc.shape[1])])
    mel = pd.DataFrame(mel, columns=[f"mel_{i}" for i in range(mel.shape[1])])

    return mfcc, stft, lpc, mel


def split_data(
    mfcc: pd.DataFrame,
    stft: pd.DataFrame,
    lpc: pd.DataFrame,
    mel: pd.DataFrame,
    other_data: pd.DataFrame,
    target: pd.Series,
    params: dict,
) -> tuple[pd.DataFrame]:
    random_state = params["random_state"]
    train_size = params["train_size"]
    val_size = params["val_size"]
    options = params["options"]
    numeric_data_selection_options = params["numeric_data_selection_options"]

    data = [other_data[numeric_data_selection_options]]
    if "mfcc" in options:
        data.append(mfcc)
    if "stft" in options:
        data.append(stft)
    if "lpc" in options:
        data.append(lpc)
    if "mel" in options:
        data.append(mel)

    data_conc = pd.concat(data, axis=1)
    X_train_pre, X_med, y_train_pre, y_med = train_test_split(
        data_conc, target, random_state=random_state, train_size=train_size
    )
    X_val_pre, X_test_pre, y_val_pre, y_test_pre = train_test_split(
        X_med, y_med, random_state=random_state, train_size=val_size / (1 - train_size)
    )

    return X_train_pre, X_val_pre, X_test_pre, y_train_pre, y_val_pre, y_test_pre


def process_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> list[pd.DataFrame]:
    scaler = MinMaxScaler()
    scaler.set_output(transform="pandas")
    label_dict = {
        "neutral": 0,
        "angry": 1,
        "happy": 2,
        "sad": 3,
        "disgust": 4,
        "fear": 5,
    }

    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.fit_transform(X_val)
    # X_test = scaler.fit_transform(X_test)

    y_train = label_to_encoded(y_train, label_dict)
    y_val = label_to_encoded(y_val, label_dict)
    y_test = label_to_encoded(y_test, label_dict)

    y_train = pd.DataFrame(y_train, columns=list(label_dict.keys()))
    y_val = pd.DataFrame(y_val, columns=list(label_dict.keys()))
    y_test = pd.DataFrame(y_test, columns=list(label_dict.keys()))

    return X_train, X_val, X_test, y_train, y_val, y_test
