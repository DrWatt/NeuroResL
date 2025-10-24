import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm

def orientation(p, q, r):
    """Return orientation of the triplet (p, q, r).
    0 --> collinear
    1 --> clockwise
    2 --> counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2

def line_intersection_point(p1, p2, q1, q2):
    """
    Check if line segments p1p2 and q1q2 intersect.
    Returns:
      - (True, (x, y)) if they intersect at a single point
      - (True, None) if they overlap (collinear)
      - (False, None) if they do not intersect
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2

    # Compute orientations
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    # General case: segments properly intersect
    if o1 != o2 and o3 != o4:
        # Solve the intersection point using line equations
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return (False, None)  # Parallel or coincident

        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4)
            - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denom
        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4)
            - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denom
        return (True, (px, py))
    return (False, None)











def polygon_vertex_creator(n,r=1):
    vertices = []
    for i in range(n):
        vertices.append([r * math.cos(2 * math.pi * i / n +0.2), r * math.sin(2 * math.pi * i / n +0.2)])

    return vertices



def random_walk_triangle(num_steps,sides):
    # Define the vertices of the polygon
    global vertices
    vertices = polygon_vertex_creator(sides)

        
    global ms
    #ms = []
    global intercept
    #intercept = []
    for i in range(len(vertices)):
        ms_temp = (vertices[i][1]-vertices[i-1][1])/(vertices[i][0]-vertices[i-1][0])
        ms.append(ms_temp)
        intercept.append(-vertices[i-1][1]*ms_temp + vertices[i-1][0])

    # Initialize the position at the centroid of the triangle
    start_position = np.mean(vertices, axis=0)
    position = start_position
    # Define arrays to store x and y positions for visualization
    x_positions = [position[0]]
    y_positions = [position[1]]
    distances_x_max = []
    distances_x_min = []
    distances_x_median = []
    distances_y_max = []
    distances_y_min = []
    distances_y_median = []
    vertices_prev = np.roll(vertices, 1, axis=0)
    rand_steps = np.random.normal(0, 1, size=(num_steps, 2))
    # Perform random walk
    for s in range(num_steps):
        # Choose a random vertex as the target
        #target_vertex = vertices[np.random.randint(len(vertices))]
        # Move halfway towards the target
        new_position = position + rand_steps[s] #(position + target_vertex) / 2
        # print(position[0]*ms[0]+intercept[0])
        #inters = False
        #v = 0
        #for j in range(len(vertices)):
        #    #print(new_position, position, "position", j)
        #    #print(vertices[j],vertices[j-1], "vertices")
        #    #print(line_intersection_point(position,new_position,vertices[j],vertices[j-1]))
        #    inters, point = line_intersection_point(position,new_position,vertices[j],vertices[j-1])
        #    if inters:
        #        v = j
        #        break
        #    else:
        #        inters = False
        intersections = [
                line_intersection_point(position, new_position, v1, v2)
                for v1, v2 in zip(vertices, vertices_prev)
                ]
        inter_flags = np.array([i[0] for i in intersections])
        points = [i[1] for i in intersections]
        #while(inters):
        #    centering = np.random.uniform()
        #    new_position = np.array([point[0] + centering*(start_position[0] - point[0]),point[1] + centering*(start_position[1] - point[1])]) 
        #    inters, point = line_intersection_point(position,new_position,vertices[v],vertices[v-1])
        if np.any(inter_flags):
            v = np.argmax(inter_flags)  # first intersection
            inters = inter_flags[v]
            point = points[v]

        # Randomly recenter until inside box (single loop)
            while inters:
                new_position = point + np.random.uniform()* (start_position - point)
                inters, point = line_intersection_point(position, new_position,
                                                    vertices[v], vertices[v-1])
        position = new_position
        # distances = [np.abs(position[1] - ms*position[0] - intercept)/math.sqrt(1+ms**2) for ms,intercept in zip(ms,intercept)]
        distances_x = np.array([position[0] - ((position[1]+intercept)/ms) for  ms,intercept in zip(ms,intercept)])
        #distances_x = np.where(np.abs(distances_x) > 1.0, distances_x % 1, distances_x)
        #distances_x = np.where(distances_x > 0,distances_x,-distances_x)
        #print(distances)
        distances_x_max.append(max(distances_x))
        distances_x_min.append(min(distances_x))
        distances_x_median.append(np.median(distances_x))

        distances_y = np.array([position[1] - (position[0]*ms+intercept) for  ms,intercept in zip(ms,intercept)])
        #print(distances)
        #distances_y = np.where(np.abs(distances_y) > 1.0, distances_y % 1, distances_y)
        #distances_y = np.where(distances_y > 0,distances_y,-distances_y)
        distances_y_max.append(max(distances_y))
        distances_y_min.append(min(distances_y))
        distances_y_median.append(np.median(distances_y))
        
        # Store the new position
        x_positions.append(position[0])
        y_positions.append(position[1])

    return x_positions, y_positions, distances_x_min, distances_x_max, distances_x_median, distances_y_min, distances_y_max, distances_y_median

# Number of steps in the random walk
num_steps = 1000
num_events = 100
sides = 5
dists = []
ms = []
intercept = []
vertices = []
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
for i in range(sides):
    plt.axline(vertices[i],slope=ms[i],c="red")
    #print(intercept[i],ms[i],vertices[i])
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
