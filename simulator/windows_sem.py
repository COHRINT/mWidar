import win32api
import win32event
import winerror

def createSemaphore(name, initial_count, max_count):
    try:
        semaphore = win32event.CreateSemaphore(None, initial_count, max_count, name)
        if semaphore == 0:
            raise Exception("Failed to create semaphore")
        return semaphore
    except winerror.error as e:
        if e.winerror == winerror.ERROR_ALREADY_EXISTS:
            return win32event.OpenSemaphore(win32event.SYNCHRONIZE, False, name)
        raise

def acquireSemaphore(semaphore):
    win32event.WaitForSingleObject(semaphore, win32event.INFINITE)

def releaseSemaphore(semaphore):
    win32event.ReleaseSemaphore(semaphore, 1)
    
def closeSemaphore(semaphore):
    win32api.CloseHandle(semaphore)

if __name__ == "__main__":
    semaphore_name = "MySemaphore"
    semaphore = create_named_semaphore(semaphore_name, 0, 1)

    # Use the semaphore for synchronization
    acquire_semaphore(semaphore)
    try:
        # Perform operations that require exclusive access
        print("Acquired semaphore")
    finally:
        release_semaphore(semaphore)
        print("Released semaphore")
