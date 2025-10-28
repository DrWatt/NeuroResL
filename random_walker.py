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

    # Avoid division by zero
    denom_safe = np.where(parallel, np.nan, denom)

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    t = t_num / denom_safe
    u = u_num / denom_safe

    mask = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    points = np.stack((px, py), axis=1)
    #print(np.where(mask)[0], points[mask])
 
    return np.where(mask)[0], points[mask]

def polygon_vertex_creator(n,r=1):
    vertices = []
    for i in range(n):
        vertices.append([r * math.cos(2 * math.pi * i / n +0.2), r * math.sin(2 * math.pi * i / n +0.2)])

    return np.array(vertices)



def random_walk_polygon(num_steps,sides):
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
        intercept.append(-vertices[i-1][0]*ms_temp + vertices[i-1][1])

    # Initialize the position at the centroid of the triangle
    start_position = np.mean(vertices, axis=0)
    p = start_position
    # Define arrays to store x and y positions for visualization
    x_positions = [p[0]]
    y_positions = [p[1]]
    distances_x_max = []
    distances_x_min = []
    distances_x_median = []
    distances_y_max = []
    distances_y_min = []
    distances_y_median = []
    vertices_prev = np.roll(vertices, 1, axis=0)
    rand_steps = np.random.normal(0, .10, size=(num_steps, 2))

    #print(vertices.shape, vertices_prev.shape)
    #print(vertices, vertices_prev)

    # Perform random walk
    for s in range(num_steps):

        _p = p + rand_steps[s] #(position + target_vertex) / 2
        
        side, xing = line_intersection_point(p, _p, vertices, vertices_prev)

        #global crossx
        #global crossy
        if len(xing):
            inters = True
            xing = xing[0]
            side = int(side) 
            #crossx.append(xing[0])
            #crossy.append(xing[1])
            x_positions.append(xing[0])
            y_positions.append(xing[1])

            epss = 1e-5

            d = (ms[side]*_p[0] - _p[1] + intercept[side])/(ms[side]*ms[side] + 1.0) # (a*x + b*y +c)/(a**a + b*b) 

            _p_temp = _p.copy()

            _p_temp[0] = _p[0] - 2 * ms[side] * d # x' = x - 2 * a * d  
            _p_temp[1] = _p[1] + 2 * d  # y' = y -2 * b * d
            stop = 0

            while inters:
                direction = [_p_temp[0] - xing[0],_p_temp[1] - xing[1]]   # Modifica con centro del poligono: se newnew è fuori potrebbe uscire anche il xing spostato
                norm = math.hypot(direction[0],direction[1])
                if norm != 0:
                    direction = [direction[0]/norm, direction[1]/norm]
                else:
                    print("NORMA EQUEAL TO 0")
                xing[0] += epss*direction[0]
                xing[1] += epss*direction[1]
                side_temp, xing_temp = line_intersection_point(xing, _p_temp, vertices, vertices_prev)
                if len(xing_temp):
                    pass
                else:
                    inters = False
                    break
                xing_temp = xing_temp[0]
                side_temp = int(side_temp[0])

                d = (ms[side_temp]*_p_temp[0] - _p_temp[1] + intercept[side_temp])/(ms[side_temp]*ms[side_temp] + 1.0)  # (a*x + b*y +c)/(a**a + b*b)

                _p_temp[0] = _p_temp[0] - 2 * ms[side_temp] * d # x' = x - 2 * a * d
                _p_temp[1] = _p_temp[1] + 2.0 * d # y' = y -2 * b * d
                
                xing = xing_temp
            _p = _p_temp


        p = _p

        distances_x = np.array([p[0] - ((p[1]+intercept)/ms) for  ms,intercept in zip(ms,intercept)])
        distances_x_max.append(max(distances_x))
        distances_x_min.append(min(distances_x))
        distances_x_median.append(np.median(distances_x))

        distances_y = np.array([p[1] - (p[0]*ms+intercept) for  ms,intercept in zip(ms,intercept)])
        distances_y_max.append(max(distances_y))
        distances_y_min.append(min(distances_y))
        distances_y_median.append(np.median(distances_y))
        
        # Store the new position
        x_positions.append(p[0])
        y_positions.append(p[1])

    return x_positions, y_positions, distances_x_min, distances_x_max, distances_x_median, distances_y_min, distances_y_max, distances_y_median

# Number of steps in the random walk
num_steps = 1000
num_events = 10
sides = 5
dists = []
ms = []
intercept = []
vertices = []
#crossy = []
#crossx =[]

for i in tqdm.tqdm(range(num_events)):
    x_positions, y_positions, distances_x_min, distances_x_max, distances_x_median, distances_y_min, distances_y_max, distances_y_median = random_walk_polygon(num_steps,sides)
    dists.append(np.array([distances_x_min, distances_x_max, distances_x_median, distances_y_min, distances_y_max, distances_y_median]).transpose())


np.save("polygon_distances_"+ str(sides)+ "_sides",np.array(dists))
#Plot the random walk in the triangle
plt.figure(figsize=(18, 6))
plt.subplot(1,3,1)
plt.plot(x_positions, y_positions, '-', marker = '+')
#plt.plot(crossx,crossy, 'ro', label="xing")
plt.plot(x_positions[0], y_positions[0], 'go')  # Starting point
plt.plot(x_positions[-1], y_positions[-1], 'ro')  # Ending point
plt.title("Random Walk in a Polygon")
plt.xlabel("X Position")
plt.ylabel("Y Position")
for i in range(sides):
    plt.axline(vertices[i],slope=ms[i],c="red")
    #plt.text(vertices[i][0], vertices[i][1] + 0.2, s = 12, text = f'{i}')
    #print(intercept[i],ms[i],vertices[i])
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(1,3,2)
plt.plot(distances_x_max)
plt.plot(distances_x_min)
plt.plot(distances_x_median)

plt.subplot(1,3,3)
plt.plot(distances_y_max)
plt.plot(distances_y_min)
plt.plot(distances_y_median)

plt.show()
