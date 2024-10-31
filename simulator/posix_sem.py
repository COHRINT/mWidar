import posix_ipc

def createSemaphore(name, initial_count, max_count):
    try:
        semaphore = posix_ipc.Semaphore(name, posix_ipc.O_CREAT, initial_value=initial_count)
        return semaphore
    except posix_ipc.ExistentialError:
        return posix_ipc.Semaphore(name)
    
def acquireSemaphore(semaphore):
    semaphore.acquire()
    
def releaseSemaphore(semaphore):
    semaphore.release()

def closeSemaphore(semaphore):
    semaphore.unlink()
