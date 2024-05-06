def getGyroData(timing, deltaTime):
    # Interpolating gyro data, using the current timing, and estimating the time jump to the current frame
    # quarternion as 6958 records --> 0:6957
    
    # Find the index of the previous and next timestamp in the dataset
    prevIndex = None
    nextIndex = None
    for i, t in enumerate(imu["time"]):
        if t <= timing:
            prevIndex = i
        if t > timing and prevIndex != None:
            nextIndex = i
            break

    # If the current timing is beyond the dataset, return the last available gyro data
    if nextIndex is None:
        # Returns last row of gyro data - change this if this does not show the headset by the end
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

# Estimate and retrieve gyro data at this timestamp
gyroData = getGyroData(timing, deltaTime)

timing += deltaTime
#print(deltaTime, timing, gyroData)

# Update the orientation estimate based on gyro data using dead reckoning filter
currentOrientation = deadReck(gyroData, deltaTime)


"""OLD render loop code"""

p0, p1, p2 = [model.vertices[i] for i in face]
p0, p1, p2 = applyRotationToVertex(p0, rotateMat), applyRotationToVertex(p1, rotateMat), applyRotationToVertex(p2, rotateMat)

n0, n1, n2 = [vertexNormals[i] for i in face]