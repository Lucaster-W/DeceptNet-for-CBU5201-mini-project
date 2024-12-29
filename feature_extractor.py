import librosa
import numpy as np
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        
    def extract_all_features(self, audio_path):
        """提取所有音频特征"""
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
        
        features = {}
        
        # 提取MFCC
        features['mfcc'] = self.extract_mfcc(audio)
        
        # 提取音高特征
        features['pitch'] = self.extract_pitch(audio)
        
        # 提取能量特征
        features['energy'] = self.extract_energy(audio)
        
        # 提取节奏特征
        features['rhythm'] = self.extract_rhythm(audio)
        
        return np.concatenate([v.flatten() for v in features.values()])
    
    def extract_mfcc(self, audio):
        return librosa.feature.mfcc(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_mfcc=self.config.N_MFCC
        )
    
    def extract_pitch(self, audio):
        return librosa.piptrack(
            y=audio,
            sr=self.config.SAMPLE_RATE
        )[0]
    
    def extract_energy(self, audio):
        return librosa.feature.rms(y=audio)
    
    def extract_rhythm(self, audio):
        onset_env = librosa.onset.onset_strength(
            y=audio, 
            sr=self.config.SAMPLE_RATE
        )
        return librosa.feature.tempogram(
            onset_envelope=onset_env, 
            sr=self.config.SAMPLE_RATE
        ) 