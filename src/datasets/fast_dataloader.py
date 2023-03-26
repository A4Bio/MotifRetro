import threading
import torch
import queue
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator, background, __doc__


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(
            local_rank
        )  # create a new cuda stream in each process
        self.local_rank = local_rank
        # self.custom_collect_fn = custom_collect_fn
        
    # @background(max_prefetch=3)
    def __iter__(self):
        self.iter = super().__iter__()
        # self.iter = BackgroundGenerator(self.iter)
        self.preload()
        return self

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():
            # avoid re-entrance or ill-conditioned thread state
            return

        # Set exit event to True for background threading stopping
        self.iter.exit_event.set()

        # Exhaust all remaining elements, so that the queue becomes empty,
        # and the thread should quit
        for _ in self.iter:
            pass

        # Waiting for background thread to quit
        self.iter.join()

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None

        # 提前将变量塞进GPU
        with torch.cuda.stream(self.stream):
            for i in range(len(self.batch[0])):
                for k in self.batch[0][i]:
                    if isinstance(self.batch[0][i][k], torch.Tensor):
                        self.batch[0][i][k] = self.batch[0][i][k].to(
                            device=self.local_rank, non_blocking=True
                        )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(
            self.stream
        )  # wait tensor to put on GPU
        batch = self.batch
        # batch = self.custom_collect_fn(self.batch)
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        # If the dataloader is to be freed, shutdown its BackgroundGenerator
        self._shutdown_background_thread()