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

def debugOutput(thing, fileName="output"):
    # A simple function to debug the output the program
    # by using a separate file
    with open(f"{fileName}.txt", "a") as file:
        file.write(thing)
        file.write("\n")
    return True

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
def quaternion2Euler(q):
    roll = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2))
    pitch = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))

    return roll, pitch, yaw

# Function to calculate the conjugate (inverse rotation) of a quaternion
def quaternionConjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

# Function to calculate the product of two quaternions
def quaternionProduct(q1, q2):
    # Unpack the first quaternion
    w1, x1, y1, z1 = q1
    # Unpack the second quaternion
    w2, x2, y2, z2 = q2
    
    # Compute the product components
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
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
conjugate = quaternionConjugate(quaternion)

# Example quaternion representations
q1 = [0.707, 0, 0.707, 0]  # Quaternion with scalar part 0.707 and imaginary parts 0, 0.707, 0
q2 = [0.5, 0.5, -0.5, 0.5]  # Another example quaternion
qProduct = quaternionProduct(q1, q2)

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

def quaternionFromAxisAngle(axis, theta):
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    
    # Compute the scalar part and the vector part of the quaternion
    scalar = np.cos(theta / 2.0)
    vector = axis * np.sin(theta / 2.0)
    
    # Create the quaternion (scalar, vector)
    return np.concatenate(([scalar], vector))

def updateOrientationDeadReckoning(gyroData, deltaTime):
    global currentQ
    # Convert gyro data (angular velocity in rad/s) to delta quaternion
    angularV = np.array([gyroData['gyroX'], gyroData['gyroY'], gyroData['gyroZ']])
    theta = np.linalg.norm(angularV) * deltaTime
    if theta > 0:
        axis = angularV / np.linalg.norm(angularV)
        deltaQ = quaternionFromAxisAngle(axis, theta)
        currentQ = quaternionProduct(currentQ, deltaQ)
    return currentQ

def updateOrientationWithAccelerometer(accData):
    global currentQ
    gravityVector = np.array([accData['acclX'], accData['acclY'], accData['acclZ']])
    gravityVectorNormal = gravityVector / np.linalg.norm(gravityVector)
    up = [0, 0, 1]  # Assuming the 'up' direction in the world frame
    
    # Calculate the necessary rotation to align the gravity vector with 'up'
    correction_quaternion = quaternionFromVectors(gravityVectorNormal, up)
    currentQ = quaternionProduct(correction_quaternion, currentQ)
    return currentQ

def quaternionFromVectors(v0, v1):
    # Normalize input vectors
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    # Compute the cross product and dot product
    cross = np.cross(v0, v1)
    dot = np.dot(v0, v1)

    # Handle the special case of vectors being exactly opposite
    if dot < -0.9999:
        # Arbitrary vector orthogonal to v0
        up = np.array([1.0, 0.0, 0.0])
        if np.abs(v0[0]) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        orthogonal = np.cross(v0, up)
        orthogonal = orthogonal / np.linalg.norm(orthogonal)
        return np.array([0.0, orthogonal[0], orthogonal[1], orthogonal[2]])

    # Compute the quaternion
    s = np.sqrt((1.0 + dot) * 2.0)
    inv_s = 1.0 / s

    return np.array([s * 0.5, cross[0] * inv_s, cross[1] * inv_s, cross[2] * inv_s])

def complementaryFilter(gyroOrient, accelOrient, alphaGyroAccel):
    # Blend gyroscope and accelerometer
    gyroAccelBlend = slerp(gyroOrient, accelOrient, alphaGyroAccel)

    return gyroAccelBlend
    
def slerp(q1, q2, t):
    # Compute the cosine of the angle between the two vectors.
    dot = np.dot(q1, q2)

    # If the dot product is negative, the quaternions have opposite handedness and slerp won't take
    # the shorter path. So invert one quaternion.
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # Clamp the dot product to be in the range of Acos()
    # This may be necessary due to floating point errors that might make dot out of range [-1, 1].
    dot = np.clip(dot, -1.0, 1.0)

    # Calculate the angles between the quaternions and the interpolation factor
    theta0 = np.arccos(dot) # angle between input vectors
    theta = theta0 * t # angle between gyroOrient and the result

    q = q2 - q1 * dot
    q = q / np.linalg.norm(q)  # Normalize vector
    
    # Compute the final interpolated quaternion
    return q1 * np.cos(theta) + q * np.sin(theta)

def quaternionToRotationMatrix(q):
    """
    Convert a quaternion into a 3x3 rotation matrix.
    """
    w, x, y, z = q
    # Compute squares of components
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # Construct the rotation matrix
    rotation_matrix = np.array([
        [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)]
    ])
    return rotation_matrix

def applyRotationToVertex(vertex, rotation_matrix):
    # Assuming vertex is a Vector object or similar with x, y, z attributes
    v = np.array([vertex.x, vertex.y, vertex.z])
    # Apply the rotation matrix
    v_rotated = np.dot(rotation_matrix, v)
    # Update the vertex coordinates
    return Vector(v_rotated[0], v_rotated[1], v_rotated[2])

print(""" --- PROBLEM 1: RENDERING --- """)

def Translate(listOfTranslates, point):
    # listOfTranslates = [3, 4, 5]
    # Means that x-coords will be translated by 3, y-coords will be translated by 4, z-coords will be translated by 5
    translation = TranslateMatrix(listOfTranslates)
    print(translation)
    return translation

width = 512
height = 512

image = Image(width, height, Color(255, 255, 255, 255))
pygame.init()

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Game Engine Render: 0')
# Init z-buffer

zBuffer = [-float('inf')] * width * height

def fractionate(top, bottom):
    return float(top / bottom)

def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
    screenX = int((x + 1.0) * width / 2.0)
    screenY = int((y + 1.0) * height / 2.0)
    
    return screenX, screenY

def getPerspectiveProjectionOLD(x, y, z):
    # Appears to create a fish-eye effect instead of a perspective shift
    """ PERSPECTIVE PROJECTION VARIANT """
    camScal = -2.2
    zp = z - 1.9
    xp = x * camScal
    yp = y * camScal

    # Calculate screen coordinates
    screenX = int(((xp / zp) + 1.0) * image.width / 2)
    screenY = int(((yp / zp) + 1.0) * image.height / 2)
    
    return screenX, screenY

def getPerspectiveProjection(x, y, z, fov=90, ar=float(width) / height, near=0.1, far=1000.0):
    # Convert field of view from degrees to radians
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    z = z - 1.9

    # Setting up the perspective projection matrix
    matrix = np.array([
        [f / ar, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])
    # Homogeneous coordinates of the point
    point = np.array([x, y, z, 1])
    # Projected point
    projected_point = np.dot(matrix, point)
    # Convert from homogeneous coordinates to screen coordinates
    screenX = int((projected_point[0] / projected_point[3] + 1.0) * width / 2.0)
    screenY = int((projected_point[1] / projected_point[3] + 1.0) * height / 2.0)
    
    return screenX, screenY

def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])

def calculateObjectCenter(vertices):
    """
    Calculate the center of the object based on a set of vertices.
    """
    # Calculate the average position of all vertices
    center = Vector(0, 0, 0)
    num_vertices = len(vertices)
    
    for vertex in vertices:
        center += vertex
    
    center /= num_vertices
    
    return center

def calculateDistance(vector1, vector2):
    """
    Calculate the Euclidean distance between two 3D points.
    """
    distance = vector1.norm() - vector2.norm()
    return distance

# Load the model # TODO ENSURE Directories align
model = Model('data/headset.obj')
model.normalizeGeometry()
# Load the half-sized model
modelHalf = Model('data/headsetHalf.obj')
modelHalf.normalizeGeometry()
# Load the quarter-sized model
modelQuar = Model('data/headsetQuarter.obj')
modelQuar.normalizeGeometry()

# Calculate face normals
faceNormals = {}
for face in model.faces:
	p0, p1, p2 = [model.vertices[i] for i in face]
	faceNormal = (p2-p0).cross(p1-p0).normalize()

	for i in face:
		if not i in faceNormals:
			faceNormals[i] = []

		faceNormals[i].append(faceNormal)

faceHalfNormals = {}
for face in modelHalf.faces:
	p0, p1, p2 = [modelHalf.vertices[i] for i in face]
	faceNormal = (p2-p0).cross(p1-p0).normalize()

	for i in face:
		if not i in faceHalfNormals:
			faceHalfNormals[i] = []

		faceHalfNormals[i].append(faceNormal)

faceQuarNormals = {}
for face in modelQuar.faces:
	p0, p1, p2 = [modelQuar.vertices[i] for i in face]
	faceNormal = (p2-p0).cross(p1-p0).normalize()

	for i in face:
		if not i in faceQuarNormals:
			faceQuarNormals[i] = []

		faceQuarNormals[i].append(faceNormal)

# Calculate vertex normals
vertexNormals = []
for vertIndex in range(len(model.vertices)):
    vertNorm = getVertexNormal(vertIndex, faceNormals)
    vertexNormals.append(vertNorm)

vertexHalfNormals = []
for vertIndex in range(len(modelHalf.vertices)):
    vertNorm = getVertexNormal(vertIndex, faceHalfNormals)
    vertexHalfNormals.append(vertNorm)

vertexQuarNormals = []
for vertIndex in range(len(modelQuar.vertices)):
    vertNorm = getVertexNormal(vertIndex, faceQuarNormals)
    vertexQuarNormals.append(vertNorm)

ModelCollection = {
    'Normal': {
       'model': model,
       'faceNormals':faceNormals,
       'vertexNormals':vertexNormals
    },
    'Half':{
       'model': modelHalf,
       'faceNormals':faceHalfNormals,
       'vertexNormals':vertexHalfNormals
    },
    'Quarter':{
       'model': modelQuar,
       'faceNormals':faceQuarNormals,
       'vertexNormals':vertexQuarNormals
    }
}

iniMagnetometer = np.array([imu['magnX'][0], imu['magnY'][0], imu['magnZ'][0]])

rate = 0
timing = 0
backgroundColor = Color(255, 255, 255, 255)  # Black as the background color
running = True
while running: # Main engine loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
    
    # Recreate the image each frame to clear it
    image = Image(width, height, backgroundColor)
    zBuffer = [-float('inf')] * width * height  # Reset the depth buffer

    pygame.display.set_caption(f'Game Engine Render: {rate}')

    # Calculate the time interval between measurements
    deltaTime = calculateDeltaTime()

    # Update orientation with gyro data (simulating getting the latest data)
    gyroData = {'gyroX': imu['gyroX'][rate % len(imu['gyroX'])],
                'gyroY': imu['gyroY'][rate % len(imu['gyroY'])],
                'gyroZ': imu['gyroZ'][rate % len(imu['gyroZ'])]}
    gyroOrient = updateOrientationDeadReckoning(gyroData, deltaTime)

    # Optionally, apply accelerometer data for tilt correction
    accData = {'acclX': imu['acclX'][rate % len(imu['acclX'])],
               'acclY': imu['acclY'][rate % len(imu['acclY'])],
               'acclZ': imu['acclZ'][rate % len(imu['acclZ'])]}
    accelOrient = updateOrientationWithAccelerometer(accData)

    # Compute final orientation using the complementary filter
    currentQ = complementaryFilter(gyroOrient, accelOrient, 0.01)

    # Convert currentQ to a rotation matrix for rendering
    rotateMat = quaternionToRotationMatrix(currentQ)

    AmbientInte = 0.1
    # Define the light direction
    lightDir = Vector(0, 0, -1)
    
    # Render the image iterating through faces
    for face in model.faces:
        vertices = [model.vertices[i] for i in face]
        normals = [vertexNormals[i] for i in face]

        # Transform vertices and normals
        transformedVertices = [applyRotationToVertex(v, rotateMat) for v in vertices]
        transformedNormals = [applyRotationToVertex(n, rotateMat) for n in normals]

        # Set to true if face should be culled
        cull = False
        
        # Transform vertices and calculate lighting intensity per vertex
        transformedPoints = []
        for p, n in zip(transformedVertices, transformedNormals):
            # Only Diffuse Lighting
            #intensity = n * lightDir
            #intensity = max(0, min(1, n * lightDir)) # Specific clamping on colours to prevent byte errors

            # Basic Ambient and Diffuse Lighting combined
            diffuse = max(np.dot(n, lightDir), 0)
            intensity = max(0, min(1, AmbientInte + (1 - AmbientInte) * diffuse))  # Combine ambient and diffuse

            n = n.normalize()

            # Intensity < 0 means light is shining through the back of the face
            # In this case, don't draw the face at all ("back-face culling")
            
            if intensity <= 0:
                cull = True # Back face culling is disabled in this version

            '''PERSPECTIVE TRANSFORM'''
            #screenX, screenY = getOrthographicProjection(p.x, p.y, p.z)
            screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z)
            colorVal = int(intensity * 255)
            transformedPoints.append(Point(screenX, screenY, p.z, Color(colorVal, colorVal, colorVal, 255)))
        
        if not cull and len(transformedPoints) == 3:
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