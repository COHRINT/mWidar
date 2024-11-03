from io import BytesIO
from PIL import Image
import numpy as np
from semaphore import Semaphore
import multiprocessing.shared_memory as shm
from multiprocessing import Lock
import signal
import sys


class SharedMemory:
    def __init__(self):
        self.image_mem = None
        self.obj_mem = None
        self.image_sem = None
        self.object_sem = None
        self.object_lock = None
        self.image_lock = None
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        self.cleanup()
        sys.exit(0)

    def create_shared_memory(self, do_objects: bool, num_objects: int):
        try:
            self.image_mem = shm.SharedMemory(name='image', create=True, size=128*128*4)
            self.image_lock = Lock()
            self.image_sem = Semaphore("/image_sem", 0, 1)

            if do_objects:
                self.obj_mem = shm.SharedMemory(name='objects', create=True, size=num_objects * 3 * 4)
                self.object_lock = Lock()
                self.object_sem = Semaphore("/object_sem", 0, 1)
            return True

        except Exception as e:
            print('Error creating shared mem objects:',e)
            return False

    def fig_to_array(self, fig):
        """Convert a Matplotlib figure to a NumPy array."""
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        return np.array(img)

    def cleanup(self):
        print("Cleaning up...")
        # Perform any necessary cleanup here
        if self.image_mem:
            self.image_mem.close()
            self.image_mem.unlink()
            self.image_mem = None
        if self.image_sem:
            del self.image_sem
            self.image_sem = None
        if self.obj_mem:
            self.obj_mem.close()
            self.obj_mem.unlink()
            self.obj_mem = None
        if self.object_sem:
            del self.object_sem
            self.object_sem = None

    def send_image(self, fig):
        try:
            with self.image_lock:
                image_array = self.fig_to_array(fig)
                image_bytes = image_array.tobytes()
                if len(image_bytes) != 128*128*4:
                    print("Image size mismatch")
                    return
                self.image_mem.buf[:len(image_bytes)] = image_bytes
                self.image_sem.release()
        except Exception as e:
            print('Error:', e)
            return

    def send_objects(self, objects, counter):
        try:
            with self.object_lock:
                for obj in objects:
                    x, y = Object.transform_coordinate(obj.getPos()[0], obj.getPos()[1], 128)  # for openCV
                    x_bytes = round(x).to_bytes(4, byteorder='little', signed=True)
                    y_bytes = round(y).to_bytes(4, byteorder='little', signed=True)
                    counter_bytes = counter.to_bytes(4, byteorder='little', signed=True)
                    start_index = obj.getID() * 3 * 4
                    self.obj_mem.buf[start_index:start_index + 4] = x_bytes
                    self.obj_mem.buf[start_index + 4:start_index + 8] = y_bytes
                    self.obj_mem.buf[start_index + 8:start_index + 12] = counter_bytes
                self.object_sem.release()
        except Exception as e:
            print('Error:', e)
            return

class Object:
    def __init__(self, x0, y0, vx0, vy0, ax0, ay0, id):
        # axis are flipped to match the image
        self.x = y0
        self.y = x0
        self.vx = vy0
        self.vy = vx0
        self.ax = ay0
        self.ay = ax0
        self.id: int = id

    def getID(self):
        return self.id

    def update(self, dt):
        self.x = self.x + self.vx * dt + 0.5 * self.ax * dt ** 2
        self.y = self.y + self.vy * dt + 0.5 * self.ay * dt ** 2
        self.vx = self.vx + self.ax * dt
        self.vy = self.vy + self.ay * dt

    def getPos(self):
        return self.x, self.y

    def getVel(self):
        return self.vx, self.vy

    def getAcc(self):
        return self.ax, self.ay

    def setPos(self, x, y):
        self.x = x
        self.y = y

    def setVel(self, vx, vy):
        self.vx = vx
        self.vy = vy

    def setAcc(self, ax, ay):
        self.ax = ax
        self.ay = ay

    # noinspection PyMethodParameters
    def parseInput(arg, iterator):
        arg = arg.replace('[', '').replace(']', '')
        arg = arg.split(',')
        return Object(float(arg[0]),
                      float(arg[1]),
                      float(arg[2]),
                      float(arg[3]),
                      float(arg[4]),
                      float(arg[5]),
                      iterator)

    def transform_coordinate(x, y, image_height):
        transformed_point = (y, image_height - x - 1)
        return transformed_point