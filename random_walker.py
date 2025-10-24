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

def line_intersection_point(p1, p2, q1, q2, tol = 1e-9):
    """
    Check if line segments p1p2 and q1q2 intersect.
    Returns:
      - (True, (x, y)) if they intersect at a single point
      - (True, None) if they overlap (collinear)
      - (False, None) if they do not intersect
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1[:, 0], q1[:, 1]
    x4, y4 = q2[:, 0], q2[:, 1]


    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    parallel = np.abs(denom) < tol

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    t = t_num / denom
    u = -u_num / denom

    mask = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    points = np.stack((px, py), axis=1)
    
    return mask, points

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

        new_position = position + rand_steps[s] #(position + target_vertex) / 2
        
        
        
        
        mask, points = line_intersection_point(position, new_position, vertices, vertices_prev)
        print(mask)
        
        if np.any(mask):
            v = np.argmax(mask)  # first intersection
            point = points[v]
            inters = True
            print(point)

        # Randomly recenter until inside box (single loop)
            while inters:

                print("start and first int",start_position,point)
                c= np.random.uniform()/2
                print("c and difference",c,start_position - point)

                new_position = point + c* (start_position - point)
                print("new pos",new_position)

                inters2, points2 = line_intersection_point(position, new_position, vertices, vertices_prev)
                print("int2 and point2",inters2,points2)

                if np.any(inters2):
                    w = np.argmax(inters2)
                    point = points2[w]
                    print("point inside second if", point)
                    inters = True
                else: 
                    inters= False
            else:
                inters= False
        position = new_position






        distances_x = np.array([position[0] - ((position[1]+intercept)/ms) for  ms,intercept in zip(ms,intercept)])
        distances_x_max.append(max(distances_x))
        distances_x_min.append(min(distances_x))
        distances_x_median.append(np.median(distances_x))

        distances_y = np.array([position[1] - (position[0]*ms+intercept) for  ms,intercept in zip(ms,intercept)])
        distances_y_max.append(max(distances_y))
        distances_y_min.append(min(distances_y))
        distances_y_median.append(np.median(distances_y))
        
        # Store the new position
        x_positions.append(position[0])
        y_positions.append(position[1])

    return x_positions, y_positions, distances_x_min, distances_x_max, distances_x_median, distances_y_min, distances_y_max, distances_y_median

# Number of steps in the random walk
num_steps = 100
num_events = 1
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
