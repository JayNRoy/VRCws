from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector
import numpy as np
import math
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

print("Reading CSV file IMUData.csv...")
csvFile = open("../IMUData.csv", "r")
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

print(""" --- PROBLEM 1: RENDERING --- """)

width = 512
height = 512
fov = 100
nearDis, farDis = 5, 750
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

def getPerspectiveProjection1(x, y, z): # TODO KEEP WORKING THIS
    """ PERSPECTIVE PROJECTION VARIANT """
    # Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
    # Instead of dropping z-coord, we use it as homogeneous factor to create perspective
    focalLen = 0.2 # Adjustable by taste
    w = 1.0 / (z / focalLen + 1)
    xp = x * w
    yp = y * w
    zp = z * w
    
    # Calculate screen coordinates
    screenX = int((xp / w + 1) * image.width / 2)
    screenY = int((yp / w + 1) * image.height / 2)
    
    return screenX, screenY

def getPerspectiveProjectionTemplate(x, y, z):
    # Represent the 3D point in homogeneous coordinates
    
    # Define the perspective matrix
    
    # Multiply the point by the perspective matrix

    # Extract the transformed coordinates and homogeneous coordinate

    # Calculate screen coordinates
    screenX = 0
    screenY = 0

    return screenX, screenY

def getPerspectiveProjection(x, y, z):
    """Perspective transformation"""
    
    near, far = 1, 500
    left, right, bottom, top = -256, 256, -256, 256
    
    # Represent the 3D point in homogeneous coordinates
    point = np.array([x, y, z, 1.0])

    # Define the perspective matrix
    focal_len = 5  # Adjustable by taste
    perspective_matrix = np.array([
        [fractionate(2*near, right-left), 0, fractionate(right+left, right-left), 0],
        [0, fractionate(2*near, top-bottom), fractionate(top+bottom, top-bottom), 0],
        [0, 0, fractionate((-1 * (far + near)), far-near), fractionate((-2 * far * near), far-near)],
        [0, 0, 1, 0]
    ])


    # Multiply the point by the perspective matrix
    transformed_point = np.dot(perspective_matrix, point)

    # Extract the transformed coordinates and homogeneous coordinate
    xp, yp, zp, wp = transformed_point

    # Calculate screen coordinates
    screenX = int((xp + 1) * width / 2.0)
    screenY = int((yp + 1) * height / 2.0)

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
running = True
while running: # Main engine loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
    
    pygame.display.set_caption(f'Game Engine Render: {rate}')
    
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
            
            screenX, screenY = getOrthographicProjection(p.x, p.y, p.z)
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