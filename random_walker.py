import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm

def polygon_vertex_creator(n,r=1):
    vertices = []
    for i in range(n):
        vertices.append([r * math.cos(2 * math.pi * i / n +0.2), r * math.sin(2 * math.pi * i / n +0.2)])

    return vertices



def random_walk_triangle(num_steps,sides):
    # Define the vertices of the polygon

    vertices = polygon_vertex_creator(sides)

        
    ms = []
    intercept = []
    for i in range(len(vertices)):
        ms_temp = (vertices[i][1]-vertices[i-1][1])/(vertices[i][0]-vertices[i-1][0])
        ms.append(ms_temp)
        intercept.append(-vertices[i-1][1]*ms_temp + vertices[i-1][0])
    # Initialize the position at the centroid of the triangle
    position = np.mean(vertices, axis=0)

    # Define arrays to store x and y positions for visualization
    x_positions = [position[0]]
    y_positions = [position[1]]
    distances_x_max = []
    distances_x_min = []
    distances_x_median = []
    distances_y_max = []
    distances_y_min = []
    distances_y_median = []
    # Perform random walk
    for _ in range(num_steps):
        # Choose a random vertex as the target
        target_vertex = vertices[np.random.randint(len(vertices))]

        # Move halfway towards the target
        position = (position + target_vertex) / 2
        # print(position[0]*ms[0]+intercept[0])
        # distances = [np.abs(position[1] - ms*position[0] - intercept)/math.sqrt(1+ms**2) for ms,intercept in zip(ms,intercept)]
        distances_x = [position[0] - ((position[1]+intercept)/ms) for  ms,intercept in zip(ms,intercept)]
        #print(distances)
        distances_x_max.append(max(distances_x))
        distances_x_min.append(min(distances_x))
        distances_x_median.append(np.median(distances_x))

        distances_y = [position[1] - (position[0]*ms+intercept) for  ms,intercept in zip(ms,intercept)]
        #print(distances)
        distances_y_max.append(max(distances_y))
        distances_y_min.append(min(distances_y))
        distances_y_median.append(np.median(distances_y))
        
        # Store the new position
        x_positions.append(position[0])
        y_positions.append(position[1])

    return x_positions, y_positions, distances_x_min, distances_x_max, distances_x_median, distances_y_min, distances_y_max, distances_y_median

# Number of steps in the random walk
num_steps = 1000
num_events = 1000
sides = 5
dists = []
for i in tqdm.tqdm(range(num_events)):
    x_positions, y_positions, distances_x_min, distances_x_max, distances_x_median, distances_y_min, distances_y_max, distances_y_median = random_walk_triangle(num_steps,sides)
    dists.append(np.array([distances_x_min, distances_x_max, distances_x_median, distances_y_min, distances_y_max, distances_y_median]).transpose())


np.save("polygon_distances_"+ str(sides)+ "_sides",np.array(dists))
#Plot the random walk in the triangle
plt.figure(figsize=(18, 6))
plt.subplot(1,3,1)
plt.plot(x_positions, y_positions, '-')
plt.plot(x_positions[0], y_positions[0], 'go')  # Starting point
plt.plot(x_positions[-1], y_positions[-1], 'ro')  # Ending point
plt.title("Random Walk in a Polygon")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(distances_x_max)
plt.plot(distances_x_min)
plt.plot(distances_x_median)

plt.subplot(1,3,3)
plt.plot(distances_y_max)
plt.plot(distances_y_min)
plt.plot(distances_y_median)

plt.show()
