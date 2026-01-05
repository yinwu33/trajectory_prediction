import torch

torch.set_float32_matmul_precision("high")  # high / medium
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

def build_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    callback_cfgs = cfg.get("callbacks")
    if not callback_cfgs:
        return []
    return [instantiate(cb_cfg, _recursive_=False) for cb_cfg in callback_cfgs]


@hydra.main(version_base=None, config_path="configs", config_name="config_vectornet")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    pl.seed_everything(cfg.seed)

    dm = instantiate(cfg.datamodule, cfg=cfg, _recursive_=False)
    model = instantiate(cfg.model, cfg=cfg, _recursive_=False)

    # print("Compiling model...")
    # model = torch.compile(model)
    # print("Model compiled.")

    profiler = None
    if cfg.trainer.profiler is not None:
        cfg.trainer.max_epochs = 3

    logger = instantiate(cfg.logger)
    logger.log_hyperparams(cfg)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=build_callbacks(cfg),
        **cfg.trainer,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.get("resume_from"))
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
