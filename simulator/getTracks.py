import numpy as np

# Define a time step for acceleration calculation
dtsamp = 1  # Assuming a unit time step; adjust if necessary

# corners and center test
Pos = np.zeros((2,115))
# start at top left corner, hold
for i in range(25):
    Pos[0,i] = -2
    Pos[1,i] = 4
# move to top right corner
for i in range(25,30):
    Pos[0,i] = -2 + 0.8*(i-25)
    Pos[1,i] = 4
# hold at top right corner
for i in range(30,55):
    Pos[0,i] = 2
    Pos[1,i] = 4
# move to bottom right corner
for i in range(55,60):
    Pos[0,i] = 2
    Pos[1,i] = 4 - 0.8*(i-55)
# hold at bottom right corner
for i in range(60,85):
    Pos[0,i] = 2
    Pos[1,i] = 0
# move to bottom left corner
for i in range(85,90):
    Pos[0,i] = 2 - 0.8*(i-85)
    Pos[1,i] = 0
# hold at bottom left corner
for i in range(90,115):
    Pos[0,i] = -2
    Pos[1,i] = 0

np.save('data/tracks/SimTraj_Corners.npy', Pos)

# Line Trajectory
Pos = np.zeros((2, 40))
for i in range(40):
    Pos[0, i] = 0.0025 * (i) ** 2 - 2
    Pos[1, i] = -0.001 * (i) ** 2 + 3.75

trueVel = np.zeros_like(Pos)
for i in range(Pos.shape[1]):
    trueVel[0, i] = 0.005 * (i)
    trueVel[1, i] = -0.002 * (i)

TrueAcc = np.zeros_like(trueVel)
TrueAcc[0, :] = 0.005
TrueAcc[1, :] = -0.002

np.save('data/tracks/SimTraj_Line.npy', Pos)
np.save('data/tracks/TrueVelLine.npy', trueVel)
np.save('data/tracks/TrueAccLine.npy', TrueAcc)

# Circle Trajectory
Pos = np.zeros((2, 100))
for i in range(100):
    Pos[0, i] = 1.5 * np.cos(i / 15 - 1)
    Pos[1, i] = 1.5 * np.sin(i / 15 - 1) + 2

trueVel = np.zeros_like(Pos)
for i in range(Pos.shape[1]):
    trueVel[0, i] = (-1.5 / 9) * np.sin(i / 9 - 1)
    trueVel[1, i] = (1.5 / 9) * np.cos(i / 9 - 1)

TrueAcc = np.zeros_like(trueVel)
for i in range(Pos.shape[1]):
    TrueAcc[0, i] = (-1.5 / 81) * np.cos(i / 9 - 1)
    TrueAcc[1, i] = (-1.5 / 81) * np.sin(i / 9 - 1)

np.save('data/tracks/SimTraj_Circle.npy', Pos)
np.save('data/tracks/TrueVelCircle.npy', trueVel)
np.save('data/tracks/TrueAccCircle.npy', TrueAcc)

# V-Shape Trajectory
Pos = np.zeros((2, 100))  # Initialize the position array with zeros (100 points)
for i in range(40):  # First 40 points
    Pos[0, i] = 0.075 * i - 2
    Pos[1, i] = -0.09 * i + 3.75
for i in range(40, 55):  # For points from 41 to 55 (no movement)
    Pos[0, i] = Pos[0, i - 1]
    Pos[1, i] = Pos[1, i - 1]
for i in range(55, 100):  # For points from 56 to 100
    Pos[0, i] = Pos[0, i - 1]
    Pos[1, i] = Pos[1, 54] + 0.085 * (i - 54)

# Compute the true velocities (constant velocities for each segment)
trueVel = np.zeros_like(Pos)
trueVel[0, :40] = 0.075  # First segment (X velocity)
trueVel[1, :40] = -0.09  # First segment (Y velocity)
trueVel[0, 40:55] = 0  # Second segment (no movement)
trueVel[1, 40:55] = 0
trueVel[0, 55:] = 0  # Third segment (no movement in X)
trueVel[1, 55:] = 0.085  # Third segment (constant Y velocity)

# Compute the true accelerations
TrueAcc = np.zeros_like(trueVel)
dtsamp = 1  # Assuming time step of 1, adjust as needed

# For points where velocity changes (at points 41 and 56)
TrueAcc[0, 40] = (trueVel[0, 40] - trueVel[0, 39]) / dtsamp
TrueAcc[1, 40] = (trueVel[1, 40] - trueVel[1, 39]) / dtsamp
TrueAcc[0, 55] = (trueVel[0, 55] - trueVel[0, 54]) / dtsamp
TrueAcc[1, 55] = (trueVel[1, 55] - trueVel[1, 54]) / dtsamp

# Save the results
np.save('data/tracks/SimTraj_Vshape.npy', Pos)
np.save('data/tracks/TrueVelVshape.npy', trueVel)
np.save('data/tracks/TrueAccVshape.npy', TrueAcc)

