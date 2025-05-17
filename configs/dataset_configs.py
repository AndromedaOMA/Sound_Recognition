from pydantic import BaseModel


class DatasetConfigs(BaseModel):
    ANNOTATIONS_FILE: str = "E:/UrbanSound8k/metadata/UrbanSound8K.csv"
    AUDIO_DIR: str = "E:/UrbanSound8k/audio"
    SAMPLE_RATE: int = 22050
    NUM_SAMPLES: int = 22050
