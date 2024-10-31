import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import json
import multiprocessing.shared_memory as shm
from multiprocessing import Lock, Condition
from semaphore import Semaphore
import signal
import sys
from io import BytesIO
from PIL import Image


image_mem = None  # Placeholder for shared memory object
obj_mem = None  # Placeholder for shared memory object
image_sem = None  # Placeholder for semaphore object
object_sem = None  # Placeholder for semaphore object

def transform_coordinate(x, y, image_height):
    transformed_point = (y, image_height - x - 1)
    return transformed_point


def fig_to_array(fig):
    """Convert a Matplotlib figure to a NumPy array."""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

def cleanup():
    print("Cleaning up...")
    # Perform any necessary cleanup here
    if image_mem:
        image_mem.close()
        image_mem.unlink()
        image_mem = None
        del image_sem
    if obj_mem:
        obj_mem.close()
        obj_mem.unlink()
        obj_mem = None
        del object_sem

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)
    


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

signal.signal(signal.SIGINT, signal_handler)
print("Press Ctrl+C to exit...")
def main():
    # register signal handler for clean up
    
    # read in matrices for sampling and recovery
    M = np.load('sampling-12tx-4096samples-128x128.npy')
    G = np.load('recovery-12tx-4096samples-128x128.npy')
    
    # read in arguments
    parser = argparse.ArgumentParser(description='Simulate object tracks.')
    parser.add_argument('-o', '--objects', nargs='+', required=True, help='List of objects in the form "[x0,y0,vx0,vy0,ax0,ay0]"')
    parser.add_argument('-t', '--time', type=float, default=0.2, help='Time step for simulation')
    parser.add_argument('-f', '--output', type=str, default="output/img", help='Output file to save the image')
    parser.add_argument('-d', '--display', type=bool, default=False, help='Setting this flag will display the image')
    parser.add_argument('-T', '--truth', type=bool, default=False, help='Export the object coordinate data (rounded to nearest integer) to a file')
    parser.add_argument('-s', '--shared', type=bool, default=False, help='Use shared memory for image data')
    args = parser.parse_args()
    
    # create objects from 
    object_id_iterator = 0
    objects = []
    for arg in args.objects:
        objects.append(parseInput(arg, object_id_iterator))
        object_id_iterator += 1
        
        
    # create shared memory for image data if needed
    if (args.shared == True):
        try:
            image_mem = shm.SharedMemory(name='image', create=True, size=128*128*4)
            image_lock = Lock()
            image_cond = Condition(image_lock)
            image_sem = Semaphore("/image_sem", 0, 1)
            object_sem = Semaphore("/object_sem", 0, 1)
            
        except Exception as e:
            print('Error creating shared mem objects:',e)
            return
        if (args.truth == True):
            try:
                obj_mem = shm.SharedMemory(name='objects', create=True, size=len(objects) * 3 * 4)
                object_lock = Lock()
                object_cond = Condition(object_lock)
            except Exception as e:
                print(e)
        
    
    # run the sim
    print('Running simulation...')
    if (args.display == True):
        plt.ion() # make the plot interactive
    fig, ax = plt.subplots(figsize=(1.28, 1.28), dpi=100)
    counter: int = 0
    
    
    
    while True:
        counter = counter + 1
        S = np.zeros((128, 128))
        for obj in objects:
            print(f"Object {obj.getID()} at: ", obj.getPos())
            if (obj.x > 0 and obj.x < 128) and (obj.y > 0 and obj.y < 128):
                S[int(obj.x)][ int(obj.y)] = 1
            signal = M.dot(S.flatten())
            obj.update(args.time)
        image = signal.dot(G).reshape(128, 128)
        
        ax.cla()  # Clear the axis for the next iteration
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        # Ensure the plot is a 128x128 square
        ax.set_aspect('equal')
        ax.set_xlim(0, 128)
        ax.set_ylim(0, 128)
        plt.axis('off')
        if (args.display == True): 
            plt.imshow(image, origin='lower')
            plt.pause(0.01)  # Pause to allow the plot to update
        else:
            c = ax.pcolormesh(image, cmap='viridis', shading='auto')
            fig.canvas.draw()
        time.sleep(args.time)
        if (args.shared == True):
            try:
                with image_lock:
                    image_array = fig_to_array(fig)
                    image_bytes = image_array.tobytes()
                    if (len(image_bytes) != 128*128*4):
                        print("Image size mismatch")
                        continue
                    image_mem.buf[:len(image_bytes)] = image_bytes
                    image_cond.notify_all()
                    image_sem.release()
                if (args.truth == True):
                    with object_lock:
                        for obj in objects:
                            x, y = transform_coordinate(obj.getPos()[0], obj.getPos()[1], 128)  # for openCV
                            x_bytes = round(x).to_bytes(4, byteorder='little', signed=True)
                            y_bytes = round(y).to_bytes(4, byteorder='little', signed=True)
                            counter_bytes = counter.to_bytes(4, byteorder='little', signed=True)
                            start_index = obj.getID() * 3 * 4
                            obj_mem.buf[start_index:start_index + 4] = x_bytes
                            obj_mem.buf[start_index + 4:start_index + 8] = y_bytes
                            obj_mem.buf[start_index + 8:start_index + 12] = counter_bytes
                        object_cond.notify_all()
                        object_sem.release()
            except Exception as e:
                print('Error:', e)
                continue
        else:
            try:
                plt.savefig(args.output + ".png", bbox_inches='tight', pad_inches=0)
                if (args.truth == True):
                    with open(args.output + "_coords.json", "w") as f:
                        objectsList = []
                        # the coordinate framing in openCV is flipped -- 0,0 is top left, whereas in numpy it is bottom left
                        for obj in objects:
                            x ,y = transform_coordinate(obj.getPos()[0], obj.getPos()[1], 128)
                            objDict = {
                                "id": obj.getID(),
                                "x": int(x),
                                "y": int(y),
                                "time": counter
                            }
                            objectsList.append(objDict)
                        json.dump(objectsList, f, indent=4) # do notinclude formatting, messes up C++ parser
            except Exception as e:
                print(e)
                continue

if __name__ == '__main__':
    main()