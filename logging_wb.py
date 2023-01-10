import wandb

from model_picker import ModelType


def init_logging(run_name: str, learning_rate: float, num_epochs: int, batch_size: int, modelType: ModelType, scale: float, model):
    wandb.init(
        project="3d-insect-classification",
        name=run_name,
        entity="ml_dtu",
        config={
            "learning rate": learning_rate,
            "number of epochs": num_epochs,
            "batch size": batch_size,
            "model": modelType.name,
            "scale": scale,
            "optimiser": "Adam"
        })

    wandb.watch(model)


def log_test_result(result, log_prefix: str):
    wandb.log({
        f"{log_prefix}loss": result.loss,
        f"{log_prefix}accuracy": result.acc,
        f"{log_prefix}area under curve mean": result.auc.mean().item()
    })