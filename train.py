import torch
from tqdm import tqdm

from configs import RunConfig
from models import BaseModel

def train(config: RunConfig):
    model_config = config.model
    env_config = config.envionment
    train_config = config.trainer
    dataset_config = config.dataset

    tqdm.write("Loading model...")

    model: BaseModel = model_config.get_model(env_config).to(train_config.device)
    if model_config.torch_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model.compile()

    train_config.get_trainer().train(model, model_config, env_config, dataset_config)