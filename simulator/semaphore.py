import sys
import os

# Import platform-specific semaphore functions
if sys.platform == 'linux' or sys.platform == 'darwin':
    from posix_sem import createSemaphore, acquireSemaphore, releaseSemaphore, closeSemaphore
elif sys.platform == 'win32':
    from windows_sem import createSemaphore, acquireSemaphore, releaseSemaphore, closeSemaphore
else:
    print("Unsupported OS")
    os._exit(1)

class Semaphore:
    def __init__(self, name, initial_count, max_count):
        try:
            self.semaphore = createSemaphore(name, initial_count, max_count)
            if self.semaphore is None:
                raise Exception("Failed to create semaphore")
        except Exception as e:
            print(f"Error creating semaphore: {e}")
            self.semaphore = None

    def acquire(self):
        if self.semaphore is not None:
            try:
                acquireSemaphore(self.semaphore)
            except Exception as e:
                print(f"Error acquiring semaphore: {e}")

    def release(self):
        if self.semaphore is not None:
            try:
                releaseSemaphore(self.semaphore)
            except Exception as e:
                print(f"Error releasing semaphore: {e}")

    def close(self):
        if self.semaphore is not None:
            try:
                closeSemaphore(self.semaphore)
                self.semaphore = None
            except Exception as e:
                print(f"Error closing semaphore: {e}")

    def __del__(self):
        self.close()