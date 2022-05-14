import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math
import copy

# discrete time
dt = 0.05

# initial velocity
u1 = -1  # [m/s]
u2 = 1   # [rad/s]

# feedback coefficients
k1, k2, k3 = [1, 2, 3]

# orientation direction
r_dire = 0.5

# target point
x_d = np.array([2.0, 1.0])  # select any decimals, not intergers

# initial point
x_init = np.array([4.0, 5.0])  # select any decimals, not intergers

count = 0

# Class for determining position of the ball
class Ball():
    def __init__(self):
        self.z = np.array([x_init[0], 0.0, x_init[1]])
        self.v1 = u1 * math.cos(math.atan(self.z[1]))
        self.v2 = u2 / (math.cos(math.atan(self.z[1])) ** 2)

    def state_update(self):
        global count
        self.z_temp = copy.copy(self.z)
        print(self.z)

        count += 1

        # You can change the starting time of z1 feedback 
        if count > 80:
            self.v1 = - k1 * (self.z[0] - x_d[0])

        if self.v1 > 0:    
            self.v2 = - k2 * self.z_temp[1] * self.v1 - k3 * (self.z_temp[2] - x_d[1]) * self.v1
        else:
            self.v2 = k2 * self.z_temp[1] * self.v1 - k3 * (self.z_temp[2] - x_d[1]) * self.v1

        self.z[0] = self.z_temp[0] + self.v1 * dt
        self.z[1] = self.z_temp[1] + self.v2 * dt
        self.z[2] = self.z_temp[2] + self.z_temp[1] * self.v1 * dt

        return self.z[0], self.z[2], self.z[1]

# Class for drawing balls
class Drawing_ball():
    def __init__(self, ax):
        self.ball_img, = ax.plot([], [], color = 'b')
        self.ax = ax

    # return (x,y) of the edge of the circles
    def draw_circle(self, center_x, center_y, balls_size, dire):
        circle_size = balls_size
        steps = 100  # enough to draw circles
        self.circle_x = []  # x of the circle
        self.circle_y = []  # y of the circle

        for j in range(steps):
            self.circle_x.append(center_x + circle_size * math.cos(j * 2 * math.pi/steps))
            self.circle_y.append(center_y + circle_size * math.sin(j * 2 * math.pi/steps))
            self.circle_x.append(center_x + r_dire * math.cos(math.atan(dire)) + 0.2 * circle_size * math.cos(j * 2 * math.pi/steps))
            self.circle_y.append(center_y + r_dire * math.sin(math.atan(dire)) + 0.2 * circle_size * math.sin(j * 2 * math.pi/steps))

        return self.circle_x, self.circle_y

    def set_graph_data(self):
        self.ball_img.set_data(self.circle_x, self.circle_y)
        return self.ball_img, 

# Function for updating animation
def update_anim(i):  # This i increases with each update
    balls_imgs = []

    balls_size.append(0.3)
    balls.append(Ball())
    balls_drawers.append(Drawing_ball(ax))
   
    # update the state of balls
    temp_x, temp_y, temp_t = balls[0].state_update()

    temp_x, temp_y = balls_drawers[0].draw_circle(temp_x, temp_y, balls_size[0], temp_t)

    balls_imgs.append(balls_drawers[0].set_graph_data())

    # time
    step_text.set_text('step = {0}'.format(i))

    return balls_imgs


#############################################################
#########################__MAIN__############################

# make figure instance
fig = plt.figure()

# make axes instance
ax = fig.add_subplot(111)  

# set the range of the axis
minmax_x = [-6, 6]
minmax_y = [-4, 8]

ax.set_xlim(minmax_x[0], minmax_x[1])
ax.set_ylim(minmax_y[0], minmax_y[1])

# equalize aspect ratio
ax.set_aspect('equal')

# set x-y axis names
ax.set_xlabel('X [m]') 
ax.set_ylabel('Y [m]') 

# other settings
ax.grid(True)  # grid-on
ax.legend()    # legend-on

# show steps
step_text = ax.text(0.05, 0.9, '', transform = ax.transAxes)

# make balls
balls_num = 1
balls_size = []
balls = []
balls_drawers = []

# draw animation
plt.scatter([x_init[0], x_d[0]], [x_init[1], x_d[1]], c = ['k', 'r'], s = 100, marker = "x")
animation = ani.FuncAnimation(fig, update_anim, interval = 50, frames = 150)

## if you want to save gif, uncomment bellow ###
animation.save('time_state.gif', writer='imagemagick')

plt.show()