from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from matrix import Matrix 
from matrixChild import TranslateMatrix, RotateMatrix, ScaleMatrix
from vector import Vector
import numpy as np
import math
import time
import pygame

def deg2rad(deg): return float((math.pi / 180)) * float(deg)

print(""" --- PROBLEM 2: TRACKING:HANDLING POSE DATA --- """)
# Recorded at a rate of 256Hz
imu = { # Y and Z are swapped here in the dataset (Z was recorded as up instead of forward) - Bear this in mind
	"time":[],
	"gyroX":[], # In rad/s
	"gyroY":[], # In rad/s
	"gyroZ":[], # In rad/s
	"acclX":[],
 	"acclY":[],
	"acclZ":[],
 	"magnX":[],
 	"magnY":[],
	"magnZ":[]
}

# Function to convert Euler angle readings to quaternions
def Euler2Quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    q = np.zeros(4)
    q[0] = cy * cr * cp + sy * sr * sp
    q[1] = cy * sr * cp - sy * cr * sp
    q[2] = cy * cr * sp + sy * sr * cp
    q[3] = sy * cr * cp - cy * sr * sp

    return q

# Function to calculate Euler angles from a quaternion
def Quaternion2Euler(q):
    roll = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2))
    pitch = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))

    return roll, pitch, yaw

# Function to calculate the conjugate (inverse rotation) of a quaternion
def QuaternionConjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

# Function to calculate the product of two quaternions
def QuaternionProduct(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

print("Reading CSV file IMUData.csv...")
csvFile = open("IMUData.csv", "r") # TODO CHANGE BEFORE SUBMISSION - IMUData.csv should be in a folder above
content = csvFile.readlines()
    #### IGNORE first line
for thing in content[1:len(content)-1]:
    thing = thing.split(",")
    for item in thing:
        if "\n" in item:
            item = float(item[0:len(item)-2]) # Remove the trailing \n and convert float
    imu["time"].append(float(thing[0]))
    imu["gyroX"].append(deg2rad(thing[1]))
    imu["gyroY"].append(deg2rad(thing[2]))
    imu["gyroZ"].append(deg2rad(thing[3]))
    imu["acclX"].append(float(thing[4]))
    imu["acclY"].append(float(thing[5]))
    imu["acclZ"].append(float(thing[6]))
    imu["magnX"].append(float(thing[7]))
    imu["magnY"].append(float(thing[8]))
    imu["magnZ"].append(float(thing[9]))
csvFile.close()

# Calculate Euler angles (roll, pitch, yaw) from gyro readings
gyroX = np.array(imu["gyroX"])
gyroY = np.array(imu["gyroY"])
gyroZ = np.array(imu["gyroZ"])
magnX = np.array(imu["magnX"])
magnY = np.array(imu["magnY"])
magnZ = np.array(imu["magnZ"])
# Normalize magnetometer readings to get a unit vector
magnitude = np.sqrt(magnX**2 + magnY**2 + magnZ**2)
magnX /= magnitude
magnY /= magnitude
magnZ /= magnitude

roll = np.arctan2(gyroY, gyroZ)
pitch = np.arctan2(-gyroX, np.sqrt(gyroY ** 2 + gyroZ ** 2))
yaw = np.arctan2(magnY, magnX)

'''del gyroX
del gyroY
del gyroZ
del magnX
del magnY
del magnZ'''

''' TESTING THE ABOVE FUNCTIONS WORK '''

# Processing data in batches
chunkz = 1000 # Chunk size
quaternion = []
# Iterate over the Euler angles in chunks
for chunkStart in range(0, len(roll), chunkz):
    chunkEnd = min(chunkStart + chunkz, len(roll))
    rollChunk = roll[chunkStart:chunkEnd]
    pitchChunk = pitch[chunkStart:chunkEnd]
    yawChunk = yaw[chunkStart:chunkEnd]

    # Process the chunk and covert the Euler angles to quaternions
    chunkResults = [Euler2Quaternion(angleRoll, anglePitch, angleYaw)
                         for angleRoll, anglePitch, angleYaw
                         in zip(rollChunk, pitchChunk, yawChunk)]
    quaternion.extend(chunkResults)

#print(f"Quaternions calculated: \n{quaternion}")

# Calculate conjugate of quaternion
conjugate = QuaternionConjugate(quaternion)

#print(f"Conjugates calculated: \n{conjugate}")

# Example quaternion representations
q1 = [0.707, 0, 0.707, 0]  # Quaternion with scalar part 0.707 and imaginary parts 0, 0.707, 0
q2 = [0.5, 0.5, -0.5, 0.5]  # Another example quaternion
qProduct = QuaternionProduct(q1, q2)

#print(f"Quaternion product of {q1} and {q2}:\n", qProduct)

print(""" --- PROBLEM 3: TRACKING: POSE CALCULATION --- """)

# Initialize the current orientation quaternion as the identity quaternion
iniQ = np.array([1, 0, 0, 0])
currentQ = iniQ

# Initialize a variable to store the timestamp of the previous measurement
timeStamp = time.time()

# Function to calculate the time interval between measurements
def calculateDeltaTime():
    global timeStamp
    currentTime = time.time()
    deltaTime = currentTime - timeStamp
    timeStamp = currentTime
    return deltaTime

def getGyroData(timing, deltaTime):
    # Interpolating gyro data, using the current timing, and estimating the time jump to the current frame
    # quarternion as 6958 records --> 0:6957
    
    # Find the index of the previous and next timestamp in the dataset
    prevIndex = None
    nextIndex = None
    for i, t in enumerate(imu["time"]):
        if t <= timing:
            prevIndex = i
        if t > timing:
            nextIndex = i
            break

    # If the current timing is beyond the dataset, return the last available gyro data
    if nextIndex is None:
        # Returns last row of gyro data - change this if this not show the headset by the end
        return imu["gyroX"][-1], imu["gyroY"][-1], imu["gyroZ"][-1]

    # Interpolate gyro data based on timestamps
    prevTime = imu["time"][prevIndex]
    nextTime = imu["time"][nextIndex]
    prevGyroX = imu["gyroX"][prevIndex]
    prevGyroY = imu["gyroY"][prevIndex]
    prevGyroZ = imu["gyroZ"][prevIndex]
    nextGyroX = imu["gyroX"][nextIndex]
    nextGyroY = imu["gyroY"][nextIndex]
    nextGyroZ = imu["gyroZ"][nextIndex]
    
    # Interpolate gyro data linearly
    interpolatedGyroX = prevGyroX + (nextGyroX - prevGyroX) * ((timing - prevTime) / (nextTime - prevTime))
    interpolatedGyroY = prevGyroY + (nextGyroY - prevGyroY) * ((timing - prevTime) / (nextTime - prevTime))
    interpolatedGyroZ = prevGyroZ + (nextGyroZ - prevGyroZ) * ((timing - prevTime) / (nextTime - prevTime))
    
    return interpolatedGyroX, interpolatedGyroY, interpolatedGyroZ
    
def deadReck(gyroData, deltaTime):
    global currentQ

    # Retrieve gyro data
    gyroX, gyroY, gyroZ = gyroData

    # Calculate the change in orientation based on gyro data and time interval
    deltaRoll = gyroX * deltaTime
    deltaPitch = gyroY * deltaTime
    deltaYaw = gyroZ * deltaTime

    deltaQ = Euler2Quaternion(deltaRoll, deltaPitch, deltaYaw)

    # Update the quaternion representing the current orientation
    currentQ = QuaternionProduct(currentQ, deltaQ)

    return currentQ

print(""" --- PROBLEM 1: RENDERING --- """)

width = 512
height = 512

### Needed?
fov = 100
nearDis, farDis = 5, 750
###

image = Image(width, height, Color(255, 255, 255, 255))
pygame.init()

# Define camera parameters
fov = 60  # Field of view
near = 0.1  # Near clipping distance
far = 100.0  # Far clipping distance

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Game Engine Render: 0')
# Init z-buffer
zBuffer = [-float('inf')] * width * height

# Load the model # TODO ENSURE Directories align
model = Model('data/headset.obj')
model.normalizeGeometry()

def fractionate(top, bottom):
    return float(top / bottom)

def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
    screenX = int((x + 1.0) * width / 2.0)
    screenY = int((y + 1.0) * height / 2.0)
    
    return screenX, screenY

def getPerspectiveProjection(x, y, z): # TODO KEEP WORKING THIS
    # Appears to create a fish-eye effect instead of a perspective shift
    """ PERSPECTIVE PROJECTION VARIANT """
    zp = z
    xp = x
    yp = y

    # Calculate screen coordinates
    screenX = int(((xp / zp) + 1.0) * image.width / 2)
    screenY = int(((yp / zp) + 1.0) * image.height / 2)
    
    return screenX, screenY

def getPerspectiveProjectionCOMPLEX(x, y, z):
    # Represent the 3D point in homogeneous coordinates
    point = Matrix([[x], [y], [z], [1]]) # w - homogeneous coordinate

    viewAngle = 30

    # Define the perspective matrix
    perspMatrix = Matrix(
         [
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 1, 0]
        ]
    )
    # Multiply the point by the perspective matrix
    result = perspMatrix.multMatrix(point)

    # Extract the transformed coordinates and homogeneous coordinate
    zp = result.elements[2][0] * math.tan(viewAngle / 2)
    xp = result.elements[0][0]
    yp = result.elements[1][0]

    # Calculate screen coordinates
    screenX = int(((xp / zp) + 1.0) * image.width / 2)
    screenY = int(((yp / zp) + 1.0) * image.height / 2)

    return screenX, screenY

def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])

# Calculate face normals
faceNormals = {}
for face in model.faces:
	p0, p1, p2 = [model.vertices[i] for i in face]
	faceNormal = (p2-p0).cross(p1-p0).normalize()

	for i in face:
		if not i in faceNormals:
			faceNormals[i] = []

		faceNormals[i].append(faceNormal)

# Calculate vertex normals
vertexNormals = []
for vertIndex in range(len(model.vertices)):
    vertNorm = getVertexNormal(vertIndex, faceNormals)
    vertexNormals.append(vertNorm)

rate = 0
timing = 0
running = True
while running: # Main engine loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
    
    pygame.display.set_caption(f'Game Engine Render: {rate}')

    # Calculate the time interval between measurements
    deltaTime = calculateDeltaTime()

    # Estimate and retrieve gyro data at this timestamp
    gyroData = getGyroData(timing, deltaTime)

    timing += deltaTime
    #print(deltaTime, timing, gyroData)

    # Update the orientation estimate based on gyro data using dead reckoning filter
    currentOrientation = deadReck(gyroData, deltaTime)
    
    # Render the image iterating through faces
    for face in model.faces:
        p0, p1, p2 = [model.vertices[i] for i in face]
        n0, n1, n2 = [vertexNormals[i] for i in face]
        
        # Define the light direction
        lightDir = Vector(0, 0, -1)
        
        # Set to true if face should be culled
        cull = False
        
        # Transform vertices and calculate lighting intensity per vertex
        transformedPoints = []
        for p, n in zip([p0, p1, p2], [n0, n1, n2]):
            intensity = n * lightDir
            
            # Intensity < 0 means light is shining through the back of the face
            # In this case, don't draw the face at all ("back-face culling")
            if intensity < 0:
                cull = True # Back face culling is disabled in this version

            #screenX, screenY = getOrthographicProjection(p.x, p.y, p.z)
            scaler = 1.7
            screenX, screenY = getPerspectiveProjection(p.x * scaler, p.y * scaler, p.z + 2.5)
            transformedPoints.append(Point(screenX, screenY, p.z, Color(intensity*255, intensity*255, intensity*255, 255)))
        
        if not cull:
            Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(image, zBuffer)
    
    # This engine will work by writing to a single image (that is being overwritten), 
    # then showing it in the buffer.
    # Constantly overwriting images to saves space. To track the status of the engine,
    # seperate images will be written alongside the buffer image to see how the engine develops.
    
    if rate % 60 == 0: # Every 60 frames
        image.saveAsPNG(f"image_frame{rate}.png")
    image.saveAsPNG("imageBuffer.png")
    buffer = pygame.image.load("images/imageBuffer.png").convert()
    
    screen.blit(buffer, (0, 0))
    pygame.display.flip()
    rate += 1
    
pygame.quit()