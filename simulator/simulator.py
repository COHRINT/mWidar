import copy
import numpy as np
import time
import json
from helper_classes import SharedMemory
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)
def print_npy_array(file_path):
    # Load the .npy file
    array = np.load(file_path)
    # Print the array
    print(array)

def convertTomWidar(obj):
    pos_vector = np.array([obj.x, obj.y, 0])
    img_size = 128
    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    translation_vector = np.array([-img_size / 2, 0, 0])
    pos_vector = np.matmul(pos_vector, Q) + translation_vector
    obj.x = int(pos_vector[0])
    obj.y = int(pos_vector[1])
    return obj

"""
Makes a copy of the list type and performs the x axis translation.
"""
def mWidarToBottomLeft(obj):
    obj = np.array(obj)
    obj[0, :] = 64 + obj[0, :]
    return obj.tolist()

"""
Matlab code uses a smaller grid scale, set it to 128x128
"""
def scale_object(obj: list, scale):
    obj = np.array(obj) * scale
    return obj.tolist()

"""
Round the object to the nearest integer for use with the 128x128 grid.
Needed for object indexing
"""
def round_object(obj: list):
    obj = np.round(obj).astype(int)
    return obj.tolist()

def edge_bound_correction(obj):
    obj = np.array(obj)
    obj[obj < 0] = 0
    obj[obj >= 128] = 127
    return obj.tolist()
"""
Read the json of arguments and return the truth flag, independent variable, and objects (usually npy arrays)
"""
def read_json(json_path):
    # Open and read the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Extract the truth flag and independent variable
    truth_flag = data.get("truth_data", False)
    independent_var = data.get("independent_var", "x")

    # 2D array to store objects and their paths
    objects = []
    for obj in data.get("objects", []):
        if obj["type"] == "npyarray":
            file_path = obj["path"]
            # Load the npy array for the given path
            pos_data = np.load(file_path)
            objects.append(pos_data)  # Store the loaded array
    for obj in objects:
        obj = list(obj)
    objects = list(objects)

    return truth_flag, independent_var, objects


"""
Load the sample data, json file, and create shared memory object.
For each object in the json file, scale it to fit the 128x128 grid.
Loop through the objects and pass the data to the shared memory object.
Terminate the loop when there are no more objects to track.
"""
def main(argpath):
    # read in matrices for sampling and recovery
    M = np.load('simulator/data/sampling-12tx-4096samples-128x128.npy')
    G = np.load('simulator/data/recovery-12tx-4096samples-128x128.npy')
    # print_npy_array('data/tracks/SimTraj_Corners.npy')
    # create objects from arguments
    truth_flag, independent_var, objects = read_json(argpath)

    # create shared memory object
    sm = SharedMemory()
    if truth_flag:
        if not sm.create_shared_memory(True, len(objects)):
            print("Error creating shared memory objects")
            return
    else:
        if not sm.create_shared_memory(False):
            print("Error creating shared memory objects")
            return

    # scale objects to fit 128x128 image
    for i, obj in enumerate(objects):
        obj = scale_object(obj, 128 / 4)
        obj = round_object(obj)  # Ensure objects are integers
        objects[i] = obj  # Update the list
    # loop through objects and pass data
    print(objects)
    counter = 0
    while True:
        if len(objects) == 0:
            print("No more objects to track")
            break
        S = np.zeros((128, 128))
        if truth_flag:
            sm.send_objects(objects, counter)
        for i, object in enumerate(objects):
            try:
                obj = mWidarToBottomLeft(object)
                obj = edge_bound_correction(obj)
                print(f"Object {i} at: (", obj[0][counter], obj[1][counter], ")")
                if 0 <= obj[0][counter] < 128 and 0 <= obj[1][counter] < 128:
                    S[obj[1][counter]][obj[0][counter]] = 1 # S is in the form [y][x]
            except IndexError as e:
                print(f"Object {i} has no more data points.", e)
                objects.pop(i)
        signal = M.dot(S.flatten())
        arr = signal.dot(G).reshape(128, 128)
        normalized_arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = cv2.flip(normalized_arr, 0)

        # send image to shared memory
        sm.send_image(img)
        # display image
        cv2.imshow('simulator', img)
        cv2.waitKey(1)
        counter += 1
        time.sleep(0.2)
    sm.cleanup()

if __name__ == '__main__':
    main("simulator/params.json")