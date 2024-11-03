#include <iostream>
#include <vector>
#include <utility>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <csignal>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>

#define BUFFER_SIZE 2 * 3 * 4
void *g_shared_mem_ptr = nullptr;
int g_shared_mem_size = 0;

void closeConnectionToSharedMemory(void *shared_mem_ptr, int size)
{
    if (shared_mem_ptr)
    {
        munmap(shared_mem_ptr, size);
        std::cout << "Shared memory unmapped" << std::endl;
    }
    std::cerr << "failed to unmap shared mem" << std::endl;
}

void closeSemaphore(const char *sem_name)
{
    sem_t *sem = sem_open(sem_name, 0);
    if (sem != SEM_FAILED)
    {
        sem_close(sem);
        sem_unlink(sem_name);
        std::cout << "Semaphore closed and unlinked" << std::endl;
    }
}

void signalHandler(int signum)
{
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    closeConnectionToSharedMemory(g_shared_mem_ptr, g_shared_mem_size);
    closeSemaphore("/object_sem");
    exit(signum);
}

void *initializeConnectionToSharedMemory(const char *name, int size)
{
    // Open the shared memory segment
    int shm_fd;
    while (shm_fd = shm_open(name, O_RDONLY, 0666), shm_fd == -1)
    {
      std::cout << "Waiting for shared mem to open" << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    // Map the shared memory segment to the process's address space
    void *shared_mem_ptr = mmap(nullptr, size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shared_mem_ptr == MAP_FAILED)
    {
        std::cerr << "Failed to map shared memory" << std::endl;
        close(shm_fd);
        return nullptr;
    }

    std::cout << "Shared memory initialized" << std::endl;
    close(shm_fd); // Close the file descriptor as it is no longer needed after mmap
    return shared_mem_ptr;
}

sem_t *createSemaphore(const char *name, int value)
{
    sem_t *sem = sem_open(name, O_CREAT, 0666, value);
    if (sem == SEM_FAILED)
    {
        std::cerr << "Failed to create semaphore" << std::endl;
        return nullptr;
    }
    std::cout << "Semaphore created" << std::endl;
    return sem;
}

std::string readDataAsString(void *shared_mem_ptr, size_t num_objects)
{
    std::string result;
    int *data = static_cast<int *>(shared_mem_ptr);
    for (size_t i = 0; i < num_objects; ++i)
    {
        int x = data[i * 3];
        int y = data[i * 3 + 1];
        int counter = data[i * 3 + 2];
        result += "Object " + std::to_string(i) + ": x=" + std::to_string(x) + ", y=" + std::to_string(y) + ", counter=" + std::to_string(counter) + "\n";
    }
    return result;
}

std::string getDataFromBuffer(void *shared_mem_ptr, int size, sem_t *sem)
{
    // Wait for the semaphore to be signaled by the Python process
    // while (sem_wait(sem) == 0)
    // {
    //     std::cout << "Waiting for semaphore signal" << std::endl;
    // }
    sem_wait(sem);
    std::cout << "getting data" << std::endl;
    // Read the data from shared memory
    return readDataAsString(shared_mem_ptr, 1);
    std::string data = static_cast<char *>(shared_mem_ptr);
    return data;
}

int main()
{
    signal(SIGINT, signalHandler);
    const char *sem_name = "/object_sem";
    const char *data_name = "/objects";
    sem_t *sem = createSemaphore(sem_name, 0);
    g_shared_mem_ptr = initializeConnectionToSharedMemory(data_name, BUFFER_SIZE);
    g_shared_mem_size = BUFFER_SIZE;
    if (!g_shared_mem_ptr)
    {
        return -1;
    }
    std::cout << "attempting to read data" << std::endl;
    while (true)
    {
        std::string data = getDataFromBuffer(g_shared_mem_ptr, BUFFER_SIZE, sem);
        std::cout << "Data: " << data << std::endl;
    }
    return 0;
}