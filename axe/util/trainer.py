import torch
from torch import Tensor


def train_step(
    feats: Tensor,
    labels: Tensor,
    model: torch.nn.Module,
    loss: torch.nn.Module,

) -> float:

    return 0


# class Trainer:
#     def __init__(
#         self,
#         log: logging.Logger,
#         model: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         loss_fn: Union[torch.nn.Module, Callable],
#         train_data: Iterable[Union[DataLoader, Dataset]],
#         test_data: Iterable[Union[DataLoader, Dataset]],
#         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
#         max_epochs: int = 10,
#         model_dir: str = "./",
#         use_gpu_if_avail: bool = False,
#         model_train_kwargs: dict[str, Any] = {},
#         model_test_kwargs: dict[str, Any] = {},
#         disable_tqdm: bool = False,
#         no_checkpoint: bool = False,
#         train_callback: Optional[Callable[[dict], dict]] = None,
#     ) -> None:
#         self.log = log
#         self.model = model
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.train_data = train_data
#         self.test_data = test_data
#         self.train_len = self.test_len = 0
#         self.scheduler = scheduler
#         self.max_epochs = max_epochs
#         self.model_dir = model_dir
#         self.use_gpu = use_gpu_if_avail
#         self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
#         self.no_checkpoint = no_checkpoint
#         self.disable_tqdm = disable_tqdm
#
#         self._early_stop_ticks = 0
#         self.device = self._check_device()
#         self.model_train_kwargs = model_train_kwargs
#         self.model_test_kwargs = model_test_kwargs
#         self.train_callback = train_callback
#
#     def _check_device(self) -> torch.device:
#         if self.use_gpu and torch.cuda.is_available():
#             device = torch.device("cuda")
#         else:
#             device = torch.device("cpu")
#
#         return device
#
#     def _train_step(
#         self,
#         label: torch.Tensor,
#         features: torch.Tensor,
#     ) -> float:
#         label = label.to(self.device)
#         features = features.to(self.device)
#         self.optimizer.zero_grad()
#         pred = self.model(features, **self.model_train_kwargs)
#         loss = self.loss_fn(pred, label)
#         loss.backward()
#         self.optimizer.step()
#
#         return loss.item()
#
#     def _train_loop(self) -> float:
#         self.model.train()
#         self.log.info(f"{self.model_train_kwargs=}")
#
#         total_loss = 0
#         batch = 0
#         if self.train_len == 0:
#             pbar = tqdm(self.train_data, ncols=80, disable=self.disable_tqdm)
#         else:
#             pbar = tqdm(
#                 self.train_data,
#                 ncols=80,
#                 total=self.train_len,
#                 disable=self.disable_tqdm,
#             )
#         for batch, (labels, features) in enumerate(pbar):
#             loss = self._train_step(labels, features)
#             if batch % (25) == 0:
#                 pbar.set_description(f"loss {loss:e}")
#             total_loss += loss
#             if self.scheduler is not None:
#                 self.scheduler.step()
#
#         if self.train_callback is not None:
#             self.model_train_kwargs = self.train_callback(self.model_train_kwargs)
#
#         if self.train_len == 0:
#             self.train_len = batch + 1
#
#         return total_loss / self.train_len
#
#     def _validation_step(self, labels: torch.Tensor, features: torch.Tensor) -> float:
#         assert self.model_test_kwargs is not None
#         with torch.no_grad():
#             labels = labels.to(self.device)
#             features = features.to(self.device)
#             pred = self.model(features, **self.model_test_kwargs)
#             test_loss = self.loss_fn(pred, labels).item()
#
#         return test_loss
#
#     def _validation_loop(self) -> float:
#         self.model.eval()
#         test_loss = 0
#         if self.test_len == 0:
#             pbar = tqdm(
#                 self.test_data, desc="test", ncols=80, disable=self.disable_tqdm
#             )
#         else:
#             pbar = tqdm(
#                 self.test_data,
#                 desc="test",
#                 ncols=80,
#                 total=self.test_len,
#                 disable=self.disable_tqdm,
#             )
#         batch = 0
#         for batch, (labels, features) in enumerate(pbar):
#             loss = self._validation_step(labels, features)
#             # if batch % (100) == 0:
#             pbar.set_description(f"test {loss:e}")
#             test_loss += loss
#
#         if self.test_len == 0:
#             self.test_len = batch + 1  # Last batch will correspond to total
#         test_loss /= batch + 1
#
#         return test_loss
#
#     def _checkpoint(self, epoch: int, loss) -> None:
#         if self.no_checkpoint:
#             return
#
#         save_pt = {
#             "epoch": epoch,
#             "model_state_dict": self.model.state_dict(),
#             "optimizer_state_dict": self.optimizer.state_dict(),
#             "loss": loss,
#         }
#         checkpoint_name = os.path.join(
#             self.checkpoint_dir, f"epoch_{epoch:02d}.checkpoint"
#         )
#         torch.save(save_pt, checkpoint_name)
#
#         return
#
#     def _save_model(self, name: str) -> None:
#         save_path = os.path.join(self.model_dir, name)
#         torch.save(self.model.state_dict(), save_path)
#
#         return
#
#     def run(self) -> None:
#         os.makedirs(self.checkpoint_dir, exist_ok=True)
#
#         with open(os.path.join(self.model_dir, "losses.csv"), "w") as fid:
#             loss_csv_write = csv.writer(fid)
#             loss_csv_write.writerow(["epoch", "train_loss", "test_loss"])
#
#         self.log.info("Initial test with random network")
#         loss_min = self._validation_loop()
#         with open(os.path.join(self.model_dir, "losses.csv"), "a") as fid:
#             write = csv.writer(fid)
#             write.writerow([0, loss_min, loss_min])
#         self.log.info(f"Initial Loss: {loss_min}")
#
#         for epoch in range(self.max_epochs):
#             self.log.info(f"Epoch: [{epoch+1}/{self.max_epochs}]")
#             train_loss = self._train_loop()
#             curr_loss = self._validation_loop()
#             self.log.info(f"Train loss: {train_loss}")
#             self.log.info(f"Test loss: {curr_loss}")
#             self._checkpoint(epoch, curr_loss)
#
#             if curr_loss < loss_min:
#                 loss_min = curr_loss
#                 self.log.info("New minmum loss, saving...")
#                 self._save_model("best.model")
#             with open(os.path.join(self.model_dir, "losses.csv"), "a") as fid:
#                 write = csv.writer(fid)
#                 write.writerow([epoch + 1, train_loss, curr_loss])
#
#         self.log.info("Training finished")
#
#         return
