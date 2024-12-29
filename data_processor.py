import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        
    def load_audio(self, file_path):
        """加载音频文件并重采样"""
        audio, sr = librosa.load(file_path, sr=self.config.SAMPLE_RATE)
        return audio, sr
        
    def augment_audio(self, audio, sr):
        """音频数据增强"""
        augmented_audio = []
        
        # 添加高斯噪声
        noise = np.random.normal(0, 0.1, audio.shape)
        augmented_audio.append(('noise', audio + noise))
        
        # 时间拉伸
        augmented_audio.append(('slow', librosa.effects.time_stretch(audio, rate=0.8)))
        augmented_audio.append(('fast', librosa.effects.time_stretch(audio, rate=1.2)))
        
        # 音高偏移
        augmented_audio.append(('pitch_up', librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=4)))
        augmented_audio.append(('pitch_down', librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=-4)))
        
        # 音量调整
        augmented_audio.append(('volume_up', audio * 1.5))
        augmented_audio.append(('volume_down', audio * 0.5))
        
        return augmented_audio
        
    def extract_features(self, audio):
        """提取音频特征"""
        # MFCC特征
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.config.SAMPLE_RATE,
            n_mfcc=self.config.N_MFCC
        )
        
        # 频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.config.SAMPLE_RATE
        )
        
        # 色度特征
        chroma = librosa.feature.chroma_stft(
            y=audio, 
            sr=self.config.SAMPLE_RATE
        )
        
        # 组合所有特征
        features = np.concatenate([
            mfccs.mean(axis=1),
            spectral_centroids.mean(axis=1),
            chroma.mean(axis=1)
        ])
        
        return features
        
    def process_dataset(self, data, augment=False):
        """处理整个数据集"""
        features_list = []
        labels = []
        
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            file_path = os.path.join(self.config.DATA_DIR, row['filename'])
            try:
                # 加载原始音频
                audio, sr = self.load_audio(file_path)
                features = self.extract_features(audio)
                features_list.append(features)
                labels.append(1 if row['Story_type'] == 'Deceptive Story' else 0)
                
                # 如果是训练集且需要数据增强
                if augment:
                    augmented_audios = self.augment_audio(audio, sr)
                    for aug_name, aug_audio in augmented_audios:
                        aug_features = self.extract_features(aug_audio)
                        features_list.append(aug_features)
                        labels.append(1 if row['Story_type'] == 'Deceptive Story' else 0)
                        
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
                
        return np.array(features_list), np.array(labels) 