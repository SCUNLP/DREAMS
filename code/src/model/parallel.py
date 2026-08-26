"""Small thread-pool helper for I/O-bound rollout batches."""

from concurrent.futures import ThreadPoolExecutor, as_completed


def iter_completed(function, jobs, worker_count):
    if worker_count <= 1:
        for job in jobs:
            yield job, function(job)
        return

    with ThreadPoolExecutor(max_workers=min(worker_count, len(jobs))) as executor:
        futures = {executor.submit(function, job): job for job in jobs}
        for future in as_completed(futures):
            yield futures[future], future.result()
