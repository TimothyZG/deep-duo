import wandb

class Logger:
    def __init__(self, project_name, config):
        wandb.init(project=project_name, config=config,settings=wandb.Settings(_service_wait=120))

    def log(self, metrics):
        wandb.log(metrics)

    def finish(self):
        wandb.finish()
