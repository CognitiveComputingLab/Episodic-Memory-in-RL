from abc import ABC, abstractmethod
import os
from typing import Generic, TypeVar

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb

from configs import DatasetConfig, EnvConfig, ModelConfig, TrainConfig
from environments.wrappers.ObsNormalise import ObsNormaliseWrapper
from environments.wrappers.RewardScale import RewardScaleWrapper
from models import BaseModel

T = TypeVar('T', bound=TrainConfig)

class BaseTrainer(ABC, Generic[T]):
    
    def __init__(self, train_config: T) -> None:
        super().__init__()
        self.train_config: T = train_config
    
    def train(
            self, 
            model: BaseModel,
            model_config: ModelConfig,
            env_config: EnvConfig,
            dataset_config: DatasetConfig):
        
        self.model = model
        self.model_config = model_config
        self.env_config = env_config
        self.dataset_config = dataset_config
        train_config = self.train_config
        self.device = train_config.device

        if train_config.checkpoint_path is not None:
            tqdm.write("Preparing checkpointing...")
            if not os.path.isdir(train_config.checkpoint_path):
                os.makedirs(train_config.checkpoint_path, exist_ok=True)

        tqdm.write("Loading data...")
        data_source = env_config.get_datasource()
        dataset = dataset_config.get_dataset(data_source, env_config.reward_scaling)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=train_config.batch_size,
            num_workers=dataset_config.workers,
            collate_fn=dataset_config.get_collate_fn(),
            pin_memory=True,
            prefetch_factor=4 if dataset_config.workers > 0 else None,
            persistent_workers=True if dataset_config.workers > 0 else False
        )
        self.iter_data = iter(dataloader)

        tqdm.write("Getting eval environment...")
        eval_env = env_config.get_env()
        eval_env = ObsNormaliseWrapper(eval_env, data_source.obs_mean(), data_source.obs_std())
        eval_env = RewardScaleWrapper(eval_env, env_config.reward_scaling)
        eval_env.reset(seed=train_config.eval_seed)

        tqdm.write("Initialising Optimiser...")
        self.optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr)

        start_batch = 0
        resume_from = None
        if train_config.resume_from is not None:
            tqdm.write(f"Resuming from {train_config.resume_from}...")
            checkpoint = torch.load(train_config.resume_from, map_location=self.device)
            model.load_state_dict(checkpoint["model_state"])
            self.optim.load_state_dict(checkpoint["optim_state"])
            start_batch = checkpoint["batch"] + 1
            #resume_from = f"{checkpoint["wandb_id"]}?_step={start_batch}"      # Email wandb to enable

        tqdm.write("Initialising wandb...")
        wandb.init(
            mode = 'online' if train_config.do_wandb else 'disabled',
            project=train_config.project_name,
            group=train_config.group_name,
            name=train_config.run_name,
            config={
                "model": model_config.__dict__,
                "env": env_config.__dict__,
                "train": train_config.__dict__,
                "dataset": dataset_config.__dict__,
            },
            resume_from=resume_from
        )

        tqdm.write("Training...")
        with tqdm(range(start_batch, train_config.batches), 
                  unit="batch",
                  initial=start_batch,
                  total=train_config.batches) as pbar:
            for i in pbar:
                loss = self.iter()

                pbar.set_postfix(loss=f"{loss:.4f}")
                wandb.log({"train/loss": loss}, step=i)

                if i % train_config.eval_every == 0:
                    state = model.get_states()
                    
                    returns = [model.sample(eval_env, train_config.target_return) for _ in range(train_config.eval_eps)]
                    mean_return = sum(returns) / len(returns)

                    pbar.write(f"Batch {i}: mean return = {mean_return:.2f}")
                    wandb.log({
                        "eval/mean_return": mean_return,
                        "eval/max_return": max(returns),
                        "eval/min_return": min(returns),
                    }, step=i)

                    model.set_states(state)

                if i % train_config.checkpoint_every == 0 and train_config.checkpoint_path is not None:
                    assert wandb.run != None
                    checkpoint = {
                        "batch": i,
                        "model_state": model.state_dict(),
                        "optim_state": self.optim.state_dict(),
                        "wandb_id": wandb.run.id
                    }
                    torch.save(checkpoint, os.path.join(train_config.checkpoint_path, "checkpoint_" + str(i) + ".pt"))
                    torch.save(checkpoint, os.path.join(train_config.checkpoint_path, "checkpoint_latest.pt"))

        wandb.finish()
        return model

    @abstractmethod
    def iter(self) -> float:
        pass