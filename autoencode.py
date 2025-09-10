from ltng.ae import VoiceAutoEncoderCLI
from ltng.cli import MyConfigCallback


if __name__ == "__main__":
    cli = VoiceAutoEncoderCLI(
        # VoiceAutoEncoder,
        # subclass_mode_model=True,
        trainer_defaults={
            "accelerator": "cpu", # Change from "gpu"
            "strategy": {
                "class_path": "lightning.pytorch.strategies.DDPStrategy",
                "init_args": {
                    "find_unused_parameters": False,
                },
            },
            "log_every_n_steps": 1,
        },
        save_config_callback=MyConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
