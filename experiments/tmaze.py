from configs import RunConfig
from datahandling.datasets.configs import EpisodeDatasetConfig
from environments.configs import XMazeConfig
from models.DTi.Config import DTi_Config
from trainers.EpisodeTrainer import EpisodeTrainConfig

def getRun(id: int) -> RunConfig:

    model_config = DTi_Config(
            context_len=5,
            layers=3,
            heads=1,
            embed_dim=32,
            time_embed='global_sinusoidal',
            titan_layers=[1],
            num_persistent=3,
            mini_batch_size=15,  # raw token count
            block_layout="mal_blocks"
        )
    
    env_config = XMazeConfig(
        max_action=1,   # always 1
        max_ep_len=10,
        reward_scaling=1,
        min_instructions=1,
        max_instructions=2,
        possible_instructions=2,
        min_corridor_len=6,
        max_corridor_len=7,
        episodes=1000,
        clip=None,
        encoding="onehot_repeat",
        instruction_every=1
    )

    train_config = EpisodeTrainConfig(
        detach_every=None,
        lr=1e-4,
        batches=1000,
        batch_size=64,
        eval_seed=42,
        eval_every=100,
        eval_eps=15,
        target_return=0,
        grad_clip=1,
        device='cuda',
        do_wandb=False,
        project_name='tmaze_new',
        group_name='DTi',
        run_name=f'run_{id}',
        checkpoint_every=100,
        checkpoint_path=f'./checkpoints/run_{id}',
        resume_from=None
    )

    dataset_config = EpisodeDatasetConfig(
        workers=4
    )

    return RunConfig(model_config, env_config, dataset_config, train_config)