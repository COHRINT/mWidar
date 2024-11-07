# mWidar

Target tracking application in partnership with Wavesens

Note: This project is in its initial development phase, so broken code may be found within the main branch.

## Build Instructions

This project depends on a working installation of OpenCV2. Installation instructions can be found [here](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html).

This project is designed to be developed with CLion, upon cloning this repository users should be able to build the files using a build button. But building and running is simple when using other editors.

We have two separate processes, a python simulator (to be replaced with hardware) and the tracking process. Building and running the tracker can be done by creating a `build` directory in the project root, and building the project by completing the following commands:
- `cmake --build .`
- `make`
Changes to the `CmakeLists.txt` in the project root requires a rebuild of the directory, often meaning developers must delete the build directory and recompute the commands above. Usage is as follows:
```
./mWidar <image_file> <truth_file>
```

To run the python simulator, create a virtual environment, and run the `setup.[bash, cmd]` within the simulator directory. Then, activate and run with:
```
cd simulator
source venv/bin/activate
python3 simulateTracks.py -o "[x,y,vx,vy,ax,ay]" ... -s true -T true
```

Note: As of now, we are depending on a truth target association, so the `t` flag must be set to true in order to work with the truth mapping function in the tracker.

## Multithreaded Pipeline

The tracker app is designed to be run in multiple threads. Specifically, each of the three major classes should be run in their own process. As of now everything is run in a single thread, just so we can get the tracker working (see `testKFShared.cpp`). Implementation will be added for `pthread` POSIX, and Windows API implementations.
