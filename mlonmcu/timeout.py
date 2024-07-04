from multiprocessing import Process, Queue


def exec_timeout(timeout, func, *args, **kwargs):
    retQueue = Queue()

    def wrap_call():
        try:
            retQueue.put(func(*args, **kwargs))
        except Exception as e:
            retQueue.put(e)

    p = Process(target=wrap_call)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1)
        if p.is_alive():
            p.kill()
            p.join()
        raise TimeoutError("Function did not complete within the timeout time")

    ret = retQueue.get()
    if isinstance(ret, Exception):
        raise ret
    return ret
