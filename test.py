from models.ELMUR import ELMUR_Config
from models.RATE import RATE_Config


if __name__ == '__main__':
    from datahandling.datasets.configs import EpisodeDatasetConfig
    from environments.configs import XMazeConfig
    from trainers.EpisodeTrainer import EpisodeTrainConfig
    from models.DTi import DTi_Config

    """
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
    """

    """
    model_config = RATE_Config(
            context_len=5,
            layers=3,
            heads=1,
            embed_dim=32,
            time_embed='global_sinusoidal',
            num_memory_tokens=3,
            mem_heads=1
        )
    """

    """
    model_config = ELMUR_Config(
        context_len=5,
        layers=3,
        heads=1,
        embed_dim=32,
        time_embed='global_sinusoidal',
        num_memory_tokens=3,
        lr = 0.8,
        elmur_layers=[1]
    )
    """
 


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
        run_name='run_0',
        checkpoint_every=1000,
        checkpoint_path=None,
        resume_from=None
    )

    dataset_config = EpisodeDatasetConfig(
        workers=4
    )

    from train import train
    model = train(model_config, env_config, train_config, dataset_config)