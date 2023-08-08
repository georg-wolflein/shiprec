import asyncio
from dask.distributed import Scheduler, Worker, Nanny, Client, SpecCluster, Security
from contextlib import AsyncExitStack
import dask
import logging
import toolz
from typing import Sequence, Optional
from dask.widgets import get_template


# dask.config.set({"logging": {"distributed": "debug", "distributed.client": "debug", "bokeh": "debug"}})


class MixedCluster(SpecCluster):
    """A cluster with CPU and GPU workers."""

    def __init__(
        self,
        name=None,
        n_cpu_workers=1,
        gpu_devices: Optional[Sequence[int]] = (),
        host=None,
        scheduler_port=0,
        silence_logs=logging.WARN,
        dashboard_address=":8787",
        loop=None,
        services=None,
        worker_services=None,
        service_kwargs=None,
        asynchronous=False,
        security=None,
        protocol=None,
        blocked_handlers=None,
        interface=None,
        scheduler_kwargs=None,
        scheduler_sync_interval=1,
        **worker_kwargs,
    ):
        self.status = None

        security = Security()
        protocol = "tcp://"
        if host is None:
            host = "127.0.0.1"

        services = services or {}
        worker_services = worker_services or {}
        worker_dashboard_address = None
        worker_class = Nanny

        worker_kwargs.update(
            {
                "host": host,
                "nthreads": 1,
                "services": worker_services,
                "dashboard_address": worker_dashboard_address,
                "dashboard": worker_dashboard_address is not None,
                "interface": interface,
                "protocol": protocol,
                "security": security,
                "silence_logs": silence_logs,
            }
        )

        scheduler = {
            "cls": Scheduler,
            "options": toolz.merge(
                dict(
                    host=host,
                    services=services,
                    service_kwargs=service_kwargs,
                    security=security,
                    port=scheduler_port,
                    interface=interface,
                    protocol=protocol,
                    dashboard=dashboard_address is not None,
                    dashboard_address=dashboard_address,
                    blocked_handlers=blocked_handlers,
                ),
                scheduler_kwargs or {},
            ),
        }

        def make_worker(options):
            return {"cls": worker_class, "options": options}

        cpu_workers = {
            f"cpu_{i}": make_worker({**worker_kwargs, "env": {"CUDA_VISIBLE_DEVICES": ""}})
            for i in range(n_cpu_workers)
        }
        gpu_workers = {
            f"gpu_{gpu}": make_worker(
                {
                    **worker_kwargs,
                    "env": {"CUDA_VISIBLE_DEVICES": f"{gpu}"},
                    "resources": {"GPU": 1},
                }
            )
            for gpu in gpu_devices
        }
        workers = {**cpu_workers, **gpu_workers}
        workers = {i: v for i, v in enumerate(workers.values())}

        super().__init__(
            name=name,
            scheduler=scheduler,
            workers=workers,
            worker=make_worker(
                {
                    **worker_kwargs,
                    "env": {"CUDA_VISIBLE_DEVICES": ""},
                }
            ),
            loop=loop,
            asynchronous=asynchronous,
            silence_logs=silence_logs,
            security=security,
            scheduler_sync_interval=scheduler_sync_interval,
        )

    def _repr_html_(self, cluster_status=None):
        cluster_status = get_template("local_cluster.html.j2").render(
            status=self.status.name,
            processes=True,
            cluster_status=cluster_status,
        )
        return super()._repr_html_(cluster_status=cluster_status)


# async def f():
#     async with Scheduler(host="127.0.0.1") as s:
#         gpu_workers = [
#             Nanny(
#                 s.address,
#                 resources={"GPU": 1},
#                 nthreads=1,
#                 env={"CUDA_VISIBLE_DEVICES": gpu},
#                 memory_limit=MEMORY_LIMIT_PER_WORKER,
#                 name=f"gpu-{gpu}",
#             )
#             for gpu in GPUS
#         ]
#         cpu_workers = [
#             Nanny(s.address, nthreads=1, memory_limit=MEMORY_LIMIT_PER_WORKER, name=f"cpu-{i}")
#             for i in range(CPU_WORKERS)
#         ]
#         workers = gpu_workers + cpu_workers

#         async with AsyncExitStack() as stack:
#             for w in workers:
#                 await stack.enter_async_context(w)
#             print(f"Started {len(gpu_workers)} GPU workers and {len(cpu_workers)} CPU workers")
#             for w in workers:
#                 await w.finished()


# if __name__ == "__main__":
#     asyncio.get_event_loop().run_until_complete(f())

if __name__ == "__main__":
    MixedCluster()
