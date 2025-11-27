import time
from typing import Any, List, Union
from gradio_client import Client
from tqdm import tqdm
import torch


def get_model_param_count(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Calculate model's total parameter count.
    If trainable_only is True, count only parameters that require gradients.
    """
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)


class MultiClient:
    """
    Simple wrapper for multiple Gradio clients.
    """

    def __init__(self, worker_addrs: List[str]) -> None:
        self.clients = [Client(addr) for addr in worker_addrs]

    def predict(self, tasks: List[List], max_retries: int = 3) -> List[Any]:
        """
        Submit tasks to multiple Gradio clients and return predictions.
        Retries failed tasks up to `max_retries`.
        """
        pbar = tqdm(total=len(tasks))
        results = {}
        retries = {i: 0 for i in range(len(tasks))}

        # Initialize jobs for the first batch of clients
        jobs = {}
        for i, client in enumerate(self.clients):
            if i < len(tasks):
                jobs[client] = (i, client.submit(*tasks[i], api_name="/predict"))

        while jobs:
            for client, (i, job) in list(jobs.items()):
                if job.done():
                    pbar.update(1)
                    del jobs[client]
                    try:
                        results[i] = job.result()
                    except Exception as e:
                        print(f"Job {i} failed with error: {e}")
                        if retries[i] < max_retries:
                            print(f"Retrying job {i}...")
                            retries[i] += 1
                            new_job = client.submit(*tasks[i], api_name="/predict")
                            jobs[client] = (i, new_job)
                        else:
                            results[i] = None

                    # Assign next task to this client if available
                    new_i = len(results) + len(jobs)
                    if new_i < len(tasks):
                        new_task = tasks[new_i]
                        new_job = client.submit(*new_task, api_name="/predict")
                        jobs[client] = (new_i, new_job)

            time.sleep(0.1)

        pbar.close()
        # Return results in the original task order
        predicts = [results[i] for i in sorted(results)]
        return predicts