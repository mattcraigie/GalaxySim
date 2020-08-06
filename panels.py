import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# This is a simple skeleton code just to give some ideas.
# It plots collisionles particles moving at random in a cubic box in the main panel
# and shows the distribution of their separations in one of two other panels.

# For reproducibility, set a seed for randomly generated inputs. Change to your favourite integer.
np.random.seed(4080)

# Set the number of spatial dimensions (at least 2)
Nd = 3

# Choose projection for the main panel
project_3d = False

# Set the number of particles to simulate
Np = 20

# Set the total number of timesteps
Nt = 100

# Set how long the animation should dispay each timestep (in milliseconds).
frame_duration = 100

# Set initial positions at random within box
position = 1-2*np.random.random((Nd,Np))

# Set the maximum drift velocity, in units of position units per timestep
v_max= 0.01

# Set initial velocities to be random fractions of the maximum
velocity = v_max*(1-2*np.random.random((Nd,Np)))

def separation(p): # Function to find separations from position vectors
    s = p[:,None,:] - p[:,:,None] # find N x N x Nd matrix of particle separations
    return np.sum(s**2,axis=0)**0.5 # return N x N matrix of scalar separations

# Set the axes on which the points will be shown
plt.ion() # Set interactive mode on
fig = figure(figsize=(12,6)) # Create frame and set size
subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,wspace=0.15,hspace=0.2)
# Create one set of axes as the left hand panel in a 1x2 grid
if project_3d:
    ax1 = subplot(121,projection='3d') # For very basic 3D projection
else:
    ax1 = subplot(121) # For normal 2D projection
xlim(-1,1)  # Set x-axis limits
ylim(-1,1)  # Set y-axis limits

# Create command which will plot the positions of the particles
if project_3d:
    points, = ax1.plot([],[],[],'o',markersize=4)  ## For 3D projection
else:
    points, = ax1.plot([],[],'o',markersize=4) ## For 2D projection

ax2 = subplot(222) # Create second set of axes as the top right panel in a 2x2 grid
xmax = 6 # Set xaxis limit
xlim(0,xmax) # Apply limit
xlabel('Separation')
ylabel('No. of particle pairs')
dx=0.2 # Set width of x-axis bins
ylim(0,0.5*dx*Np**2) # Reasonable guess for suitable yaxis scale    
xb = np.arange(0,xmax+dx,dx)  # Set x-axis bin edges
xb[0] = 1e-6 # Shift first bin edge by a fraction to avoid showing all the zeros (a cheat, but saves so much time!)
line, = ax2.plot([],[],drawstyle='steps-post') # Define a command that plots a line in this panel

ax4 = plt.subplot(224) # Create last set of axes as the bottom right panel in a 2x2 grid

# Define procedure to update positions at each timestep
def update(i):
    global position,velocity # Get positions and velocities
    position += velocity # Increment positions according to their velocites
    points.set_data(position[0,:], position[1,:]) # Show 2D projection of first 2 position coordinates
    if project_3d:
        points.set_3d_properties(position[2,:])  ## For 3D projection
    h,x = histogram(np.ravel(tril(separation(position))),bins=xb) # Make histogram of the lower triangle of the seperation matrix
    line.set_data(x[:-1],h) # Set the new data for the line in the 2nd panel      
    return points,line, # Plot the points and the line
    
# Create animation
# https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html
ani = animation.FuncAnimation(fig, update, frames=Nt,interval = frame_duration)
#plt.show()

ani.save("panels.mp4")




