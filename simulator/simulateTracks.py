import copy

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import json
from helper_classes import Object, SharedMemory
import cv2

def convertTomWidar(obj):
    pos_vector = np.array([obj.x, obj.y, 0])
    img_size = 128
    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    translation_vector = np.array([-img_size / 2, 0, 0])
    pos_vector = np.matmul(pos_vector, Q) + translation_vector
    obj.x = int(pos_vector[0])
    obj.y = int(pos_vector[1])
    return obj

def swapAxs(obj):
    temp = obj.x
    obj.x = obj.y
    obj.y = temp
    temp = obj.vx
    obj.vx = obj.vy
    obj.vy = temp
    temp = obj.ax
    obj.ax = obj.ay
    obj.ay = temp
    return obj




def main():
    sm = None
    # read in matrices for sampling and recovery
    M = np.load('data/sampling-12tx-4096samples-128x128.npy')
    G = np.load('data/recovery-12tx-4096samples-128x128.npy')
    
    # read in arguments
    parser = argparse.ArgumentParser(description='Simulate object tracks.')
    parser.add_argument('-o', '--objects', nargs='+', required=True, help='List of objects in the form "[x0,y0,vx0,vy0,ax0,ay0]"')
    parser.add_argument('-t', '--time', type=float, default=0.2, help='Time step for simulation')
    parser.add_argument('-f', '--output', type=str, default="data/img", help='Output file to save the image')
    parser.add_argument('-d', '--display', type=bool, default=False, help='Setting this flag will display the image')
    parser.add_argument('-T', '--truth', type=bool, default=False, help='Export the object coordinate data (rounded to nearest integer) to a file')
    parser.add_argument('-s', '--shared', type=bool, default=False, help='Use shared memory for image data')
    args = parser.parse_args()
    
    # create objects from 
    object_id_iterator = 0
    objects = []
    for arg in args.objects:
        objects.append(Object.parseInput(arg, object_id_iterator))
        object_id_iterator += 1
        
    if (args.shared == True):
        sm = SharedMemory()
        if not sm.create_shared_memory(True, len(objects)):
            print("Error creating shared memory objects")
            return
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
            print(f"Object {obj.getID()} at: (", convertTomWidar(copy.deepcopy(obj)).x, convertTomWidar(copy.deepcopy(obj)).y, ") (mWidar)")
            print(f"Object {obj.getID()} at: (", obj.x, obj.y, ") (sim)")
            if (0 < obj.y < 128) and (0 < obj.x < 128):
                S[int(obj.y)][ int(obj.x)] = 1
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
                # sm.send_image(fig)
                # convert fig to opencv image
                # image_array = sm.experimental_fig_to_array(fig)
                im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('image', im)
                sm.send_cv_image(im)
                if (args.truth == True):
                    # convert objects to mWidar frame
                    objects_cpy = copy.deepcopy(objects)
                    for i in range(len(objects_cpy)):
                        objects_cpy[i] = convertTomWidar(objects_cpy[i])
                    sm.send_objects(objects_cpy, counter)
            except Exception as e:
                print(e)
                continue
        else:
            try:
                plt.savefig(args.output + ".png", bbox_inches='tight', pad_inches=0)
                if (args.truth == True):
                    with open(args.output + "_coords.json", "w") as f:
                        objectsList = []
                        # the coordinate framing in openCV is flipped -- 0,0 is top left, whereas in numpy it is bottom left
                        for obj in objects:
                            x ,y = Object.transform_coordinate(obj.getPos()[0], obj.getPos()[1], 128)
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