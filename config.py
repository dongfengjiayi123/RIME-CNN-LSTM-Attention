"""
全局配置文件
"""
import os

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== 数据配置 ==========
# 支持的数据集：taobao, tmall, cikm19
DATASET = "taobao"
DATA_PATH = os.path.join(DATA_DIR, f"{DATASET}.csv")

# 用户行为序列最大长度
MAX_SEQ_LEN = 50
# 行为类型：0=点击, 1=收藏, 2=加购, 3=购买
NUM_BEHAVIOR_TYPES = 4
# 最小行为次数过滤
MIN_USER_BEHAVIORS = 5

# ========== 模型配置 ==========
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
CNN_KERNEL_SIZE = 3
CNN_NUM_FILTERS = 64
ATTENTION_DIM = 64

# ========== 训练配置 ==========
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5  # early stopping
DEVICE = "cuda"  # 自动 fallback 到 cpu
SEED = 42

# ========== 评估配置 ==========
METRICS = ["f1", "auc", "accuracy", "precision", "recall"]
