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
from reproducibility import seed_worker
from trainers.TrainingContracts import OptimizationPolicy, TrainingExtension

T = TypeVar('T', bound=TrainConfig)

class BaseTrainer(ABC, Generic[T]):
    
    def __init__(
            self,
            train_config: T,
            extensions: tuple[TrainingExtension, ...] = ()) -> None:
        super().__init__()
        self.train_config: T = train_config
        self.training_extensions: list[TrainingExtension] = []
        for extension in extensions:
            self.add_training_extension(extension)

    def add_training_extension(self, extension: TrainingExtension) -> None:
        """Attach an optional feature without changing the base training pipeline."""
        extension.attach(self)
        self.training_extensions.append(extension)

    def remove_training_extension(self, extension: TrainingExtension) -> None:
        self.training_extensions.remove(extension)

    def _init_training(self):
        
        if self.train_config.checkpoint_path is not None:
            tqdm.write("Preparing checkpointing...")
            if not os.path.isdir(self.train_config.checkpoint_path):
                os.makedirs(self.train_config.checkpoint_path, exist_ok=True)

        tqdm.write("Loading data...")
        data_source = self.env_config.get_datasource()
        dataset = self.dataset_config.get_dataset(data_source, self.env_config.reward_scaling)
        data_generator = torch.Generator()
        data_generator.manual_seed(self.train_config.seed)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.train_config.batch_size,
            num_workers=self.dataset_config.workers,
            collate_fn=self.dataset_config.get_collate_fn(),
            worker_init_fn=seed_worker,
            generator=data_generator,
            pin_memory=True,
            prefetch_factor=4 if self.dataset_config.workers > 0 else None,
            persistent_workers=True if self.dataset_config.workers > 0 else False
        )
        self.iter_data = iter(dataloader)

        tqdm.write("Getting eval environment...")
        eval_env = self.env_config.get_env()
        eval_env = ObsNormaliseWrapper(eval_env, data_source.obs_mean(), data_source.obs_std())
        eval_env = RewardScaleWrapper(eval_env, self.env_config.reward_scaling)
        eval_env.reset(seed=self.train_config.eval_seed)
        self.eval_env = eval_env

        tqdm.write("Initialising Optimiser...")
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.lr)
        self.optimization = OptimizationPolicy(self.optim, self.train_config.grad_clip)

        start_batch = 0
        resume_from = None
        if self.train_config.resume_from is not None:
            tqdm.write(f"Resuming from {self.train_config.resume_from}...")
            checkpoint = torch.load(self.train_config.resume_from, map_location=self.device)
            format_version = checkpoint.get("runtime_state_format")
            if format_version not in (None, "episodic-memory-in-rl-v1"):
                raise ValueError(
                    f"Incompatible runtime-state format: {format_version!r}"
                )
            self.model.load_state_dict(checkpoint["model_state"])
            self.optim.load_state_dict(checkpoint["optim_state"])
            if "runtime_state" in checkpoint:
                self.model.set_states(checkpoint["runtime_state"])
            extension_state = checkpoint.get("extension_state", {})
            for extension in self.training_extensions:
                name = type(extension).__name__
                if name in extension_state:
                    extension.restore_checkpoint(extension_state[name])
            start_batch = checkpoint["batch"] + 1
            #resume_from = f"{checkpoint["wandb_id"]}?_step={start_batch}"      # Email wandb to enable
        self.start_batch = start_batch

        tqdm.write("Initialising wandb...")
        wandb.init(
            mode = 'online' if self.train_config.do_wandb else 'disabled',
            project=self.train_config.project_name,
            group=self.train_config.group_name,
            name=self.train_config.run_name,
            config={
                "model": self.model_config.__dict__,
                "env": self.env_config.__dict__,
                "train": self.train_config.__dict__,
                "dataset": self.dataset_config.__dict__,
            },
            resume_from=resume_from
        )

    
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

        self._init_training()

        tqdm.write("Training...")
        with tqdm(range(self.start_batch, train_config.batches), 
                  unit="batch",
                  initial=self.start_batch,
                  total=train_config.batches) as pbar:
            for i in pbar:
                for extension in self.training_extensions:
                    extension.before_batch(self)
                loss = self.iter()
                for extension in self.training_extensions:
                    extension.after_batch(self, loss)

                pbar.set_postfix(loss=f"{loss:.4f}")
                wandb.log({"train/loss": loss}, step=i)

                if i % train_config.eval_every == 0:
                    state = model.get_states()
                    
                    returns = [
                        model.sample(self.eval_env, train_config.target_return)
                        for _ in range(train_config.eval_eps)
                    ]
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
                        "wandb_id": wandb.run.id,
                        "runtime_state": model.get_states(),
                        "runtime_state_format": "episodic-memory-in-rl-v1",
                        "extension_state": {
                            type(extension).__name__: extension.checkpoint_state()
                            for extension in self.training_extensions
                        },
                        "seed": train_config.seed,
                        "config": {
                            "model": model_config.__dict__,
                            "env": env_config.__dict__,
                            "train": train_config.__dict__,
                            "dataset": dataset_config.__dict__,
                        },
                    }
                    torch.save(checkpoint, os.path.join(train_config.checkpoint_path, "checkpoint_" + str(i) + ".pt"))
                    torch.save(checkpoint, os.path.join(train_config.checkpoint_path, "checkpoint_latest.pt"))

        wandb.finish()
        return model

    @abstractmethod
    def iter(self) -> float:
        pass