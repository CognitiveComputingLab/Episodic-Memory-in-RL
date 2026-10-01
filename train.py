from tqdm import tqdm

from configs import ModelConfig, EnvConfig, TrainConfig, DatasetConfig

from models import BaseModel


def train(
    model_config: ModelConfig,
    env_config: EnvConfig,
    train_config: TrainConfig,
    dataset_config: DatasetConfig,
) -> BaseModel:

    tqdm.write("Loading model...")
    model: BaseModel = model_config.get_model(env_config).to(train_config.device)
    train_config.get_trainer().train(model, model_config, env_config, dataset_config)

    return model