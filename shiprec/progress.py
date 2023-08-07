"""TQDM progress bar for distributed Dask futures.

Based on https://github.com/tqdm/tqdm/issues/278#issuecomment-507006253.
"""

from tqdm.auto import tqdm as tqdm_auto
from distributed.utils import LoopRunner
from distributed.client import futures_of
from distributed.diagnostics.progressbar import ProgressBar


class TqdmNotebookProgress(ProgressBar):
    def __init__(
        self,
        keys,
        scheduler=None,
        interval="100ms",
        loop=None,
        complete=True,
        start=True,
        tqdm_class=tqdm_auto,
        **tqdm_kwargs
    ):
        self._loop_runner = loop_runner = LoopRunner(loop=loop)
        super().__init__(keys, scheduler, interval, complete)
        self.tqdm = tqdm_class(keys, **tqdm_kwargs)

        if start:
            loop_runner.run_sync(self.listen)

    def _draw_bar(self, remaining, all, **kwargs):
        update_ct = (all - remaining) - self.tqdm.n
        self.tqdm.update(update_ct)

    def _draw_stop(self, **kwargs):
        self.tqdm.close()


def tqdm_dask(futures, **kwargs):
    futures = futures_of(futures)
    if not isinstance(futures, (set, list)):
        futures = [futures]
    return TqdmNotebookProgress(futures, **kwargs)
