import os

class Config:
    def __init__(self):
        # 项目路径配置
        try:
            # 尝试使用__file__（在Python脚本中运行时）
            self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            # 在Jupyter Notebook中运行时
            self.ROOT_DIR = os.getcwd()  # 获取当前工作目录
            
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'stage2')
        self.MODELS_DIR = os.path.join(self.ROOT_DIR, 'models')
        self.FEATURES_DIR = os.path.join(self.ROOT_DIR, 'features')

        # 音频处理参数
        self.SAMPLE_RATE = 16000
        self.FRAME_LENGTH = 2048
        self.HOP_LENGTH = 512

        # 特征提取参数
        self.N_MELS = 128
        self.N_MFCC = 40

        # 模型参数
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.LEARNING_RATE = 0.001

        # 数据集参数
        self.METADATA_FILE = 'CBU0521DD_stories_attributes.csv'

        # 创建必要的目录
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.FEATURES_DIR, exist_ok=True)
        
        # 打印路径信息以便调试
        print(f"ROOT_DIR: {self.ROOT_DIR}")
        print(f"DATA_DIR: {self.DATA_DIR}") 