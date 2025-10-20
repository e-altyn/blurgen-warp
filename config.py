from dataclasses import dataclass, field


@dataclass
class TrainAugConfig:
    num_workers: int = 4
    batch_size: int = 32
    patch_size: int = 256
    data_range: float = 2.0
    
    num_poses: int = 16
    
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1000
    val_portion: float = 0.1
    
    checkpoint_dir: str = ".checkpoints"
    val_freq: int = 5
    train_seed: int = 42
    
    nafnet_grid_params: dict = field(default_factory=lambda: {
        'width': 8, 
        'middle_blk_num': 1, 
        'enc_blk_nums': [1, 1, 1, 28],
        'dec_blk_nums': [1, 1, 1, 1],
        'num_poses': 16
    })

    nafnet_comp_params: dict = field(default_factory=lambda: {
        'width': 16, 
        'middle_blk_num': 1, 
        'enc_blk_nums': [1, 1, 1, 28],
        'dec_blk_nums': [1, 1, 1, 1]
    })
