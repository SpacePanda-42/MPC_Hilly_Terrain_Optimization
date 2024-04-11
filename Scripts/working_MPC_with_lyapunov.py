
from functools import partial
from time import time

import cvxpy as cvx

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

from tqdm.auto import tqdm

def terrain(x,y):
    """
    Compute a jnp array corresponding to elevation for each (x,y) coordinate on the map

    Kwargs:
        s: The current state (jnp array)
        u: The current control input (jnp(arr))

    Returns:
        The elevation at the state s
    """
    
    mu1 = jnp.array([6, 6])
    sigma1 = jnp.array([[2, 0.5], [0.5, 2]])

    mu2 = jnp.array([6,6])
    sigma2 = jnp.array([[3, 2], [2, 3]])

    mu3 = jnp.array([5,5])
    sigma3 = jnp.array([[5,1],[1,5]])

    mu4 = jnp.array([0,10])
    sigma4 = jnp.array([[8,2],[2,4]])

    mu5 = jnp.array([9,1])
    sigma5 = jnp.array([[3, 1], [1,3]])

    mu6 = jnp.array([6,10])
    sigma6 = jnp.array([[2, 0.5], [0.5, 2]])

    mu7 = jnp.array([0,0])
    sigma7 = jnp.array([[5, 3], [3,8]])

    mu8 = jnp.array([7,4])
    sigma8 = jnp.array([[5,3],[3,8]])

    mu9 = jnp.array([5,0])
    sigma9 = jnp.array([[3,1],[1,3]])

    s = jnp.array([x,y]).T
    hill_1 = 3/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(sigma1)))*jnp.exp(-1/2*(s-mu1).T @ jnp.linalg.inv(sigma1) @ (s-mu1))
    hill_2 = 2/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(sigma2)))*jnp.exp(-1/2*(s-mu2).T @ jnp.linalg.inv(sigma2) @ (s-mu2))
    hill_3 = 3/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(sigma3)))*jnp.exp(-1/2*(s-mu3).T @ jnp.linalg.inv(sigma3) @ (s-mu3))
    hill_4 = 10/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(sigma4)))*jnp.exp(-1/2*(s-mu4).T @ jnp.linalg.inv(sigma4) @ (s-mu4))
    hill_5 = 4/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(sigma5)))*jnp.exp(-1/2*(s-mu5).T @ jnp.linalg.inv(sigma5) @ (s-mu5))
    hill_6 = 3/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(sigma6)))*jnp.exp(-1/2*(s-mu6).T @ jnp.linalg.inv(sigma6) @ (s-mu6))
    hill_7 = 5/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(sigma7)))*jnp.exp(-1/2*(s-mu7).T @ jnp.linalg.inv(sigma7) @ (s-mu7))
    hill_8 = 3/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(sigma8)))*jnp.exp(-1/2*(s-mu8).T @ jnp.linalg.inv(sigma8) @ (s-mu8))
    hill_9 = 1/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(sigma9)))*jnp.exp(-1/2*(s-mu9).T @ jnp.linalg.inv(sigma9) @ (s-mu9))

    
    return hill_1 + hill_2 + hill_3 + hill_4 + hill_5 + hill_6 + hill_7 + hill_8 + hill_9



def inclination(x,y,θ):
    """
    Compute the inclination of the car at a given (x,y) point at a given heading

    Kwargs: 
        terrain: function that returns elevation for a given location
        s: state (array or array-like)
        u: control (array or array-like)
    
    Returns:
        None

    """
    dfdx,dfdy = jax.jacrev(terrain,argnums = 0)(x,y),jax.jacrev(terrain,argnums = 1)(x,y)
    cosθ,sinθ = jnp.cos(θ),jnp.sin(θ)
    β = jnp.arctan(dfdx*cosθ + dfdy*sinθ)
    return β

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`."""

    A, B = jax.jacfwd(f, (0,1))(s,u)
    c = f(s,u) - A @ s - B @ u

    return A, B, c


def signed_distances(s, centers, radii):
    """Compute signed distances to circular obstacles."""
    centers = jnp.reshape(centers, (-1, 2))
    radii = jnp.reshape(radii, (-1,))
    
    d = jnp.linalg.norm(s[:2] - centers, axis=1) - radii

    return d


def dynamics(s, u, dt = 0.05):
    """
    Propagate the state forward by one time step

    Kwargs:
        Terrain: The elevation function that accepts x,y as inputs and returns the elevation, inclination = function that accepts coordinates x,y and a heading and returns the angle of inclination, s = state (x,y,β,θ), u = control (v, dθdt)
        Inclination: The inclination function 
        s: The current state (jnp array)
        u: The current control (jnp array)
        dt: Time step value (float)

    Returns:
        s_next: The state propagated forward by one time step
    
    """
    v,dθdt = u
    x,y,θ = s
    β = inclination(x,y,θ)
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    cosβ = jnp.cos(β)
    dxdt = v*cosθ*cosβ
    dydt = v*sinθ*cosβ
    
    x_next = x + dt*dxdt
    y_next = y + dt*dydt
    θ_next = θ + dt*dθdt
    β_next = inclination(x_next,y_next,θ_next) #function to calculate inclination at next location

    
    s_next = jnp.array([x_next,y_next,θ_next])
    return s_next


def energy(s,u):
    """
    Compute normalized energy efficiency for given state and control input
    
    Kwargs:
        s: current state (jnp array)
        u: current control (jnp array)
    
    Returns:
        1/η: The normalized energy efficiency for given (s,u) (float)
    """
    v,_ = u
    x,y,θ = s
    β = inclination(x,y,θ)
    
    #Efficiency wrt speed
    s = jnp.abs(v) #speed in mph
    A = jnp.where(s<=5,-60*s + 500.,0.)
    B = jnp.where(s>5.,0.9*(s**2) - 24.5*s + 300,0.)
    C = jnp.where(s>15.,0.0397047397*(s**2) - 0.422299922*s + 132.400932 - (0.9*(s**2) - 24.5*s + 300),0.)
    Whpmi = A + B + C
    #Normalize wrt 15 mph
    Wn = Whpmi/135.
    #Now invert to get efficiency estimate
    η_v = 1/Wn
 
    #Normalize wrt 15 mph
    Wn = Whpmi/135.
    #Now invert to get efficiency estimate
    η_v = 1/Wn
    #Efficiency wrt inclination
    a = -1.0896759704238018e+05
    b = -644.0164927921546
    c = 586.443606547018
    d = -31.441192587147885
    e = -0.012009213842952293
    η_i = jnp.exp(a*(β**4) + b*(β**3) + c*(β**2) + d*β + e) + 0.01 #extra 1% to prevent from blowing up
    
    η = η_v*η_i #net efficiency
    # print(η)
    return 1/η #normalized energy to 1


def energy_for_plotting(s,u):
    """
    Compute normalized energy efficiency for given state and control input
    
    Kwargs:
        s: current state (jnp array)
        u: current control (jnp array)
    
    Returns:
        1/η: The normalized energy efficiency for given (s,u) (float)
    """
    v,_ = u
    x,y,θ = s
    β = inclination(x,y,θ)
    
    #Efficiency wrt speed
    s = jnp.abs(v) #speed in mph
    A = jnp.where(s<=5,-60*s + 500.,0.)
    B = jnp.where(s>5.,0.9*(s**2) - 24.5*s + 300,0.)
    C = jnp.where(s>15.,0.0397047397*(s**2) - 0.422299922*s + 132.400932 - (0.9*(s**2) - 24.5*s + 300),0.)
    Whpmi = A + B + C
    #Normalize wrt 15 mph
    Wn = Whpmi/135.
    #Now invert to get efficiency estimate
    η_v = 1/Wn
 
    #Normalize wrt 15 mph
    Wn = Whpmi/135.
    #Now invert to get efficiency estimate
    η_v = 1/Wn
    #Efficiency wrt inclination
    a = -1.0896759704238018e+05
    b = -644.0164927921546
    c = 586.443606547018
    d = -31.441192587147885
    e = -0.012009213842952293
    η_i = jnp.exp(a*(β**4) + b*(β**3) + c*(β**2) + d*β + e) + 0.01 #extra 1% to prevent from blowing up
    
    η = η_v*η_i #net efficiency
    # print(η)
    return η #normalized energy to 1

def scp_iteration(f, s0, s_goal, s_prev, u_prev, P, Q, R,α):
    """Solve a single SCP sub-problem for the obstacle avoidance problem."""
    n = s_prev.shape[-1]    # state dimension
    m = u_prev.shape[-1]    # control dimension
    N = u_prev.shape[0]     # number of steps
    np.save("s_prev.npy",s_prev)
    np.save("u_prev.npy",u_prev)
    Af, Bf, cf = affinize(f, s_prev[:-1], u_prev)
    Af, Bf, cf = np.array(Af), np.array(Bf), np.array(cf)
    Ae,Be,ce = affinize(energy, s_prev[:-1], u_prev)
    Ae, Be, ce = np.array(Ae), np.array(Be), np.array(ce)
    
    Ad, _, cd = affinize(lambda s, _: d(s), s_prev,
                         jnp.concatenate((u_prev, u_prev[-1:])))
    Ad, cd = np.array(Ad), np.array(cd)
    
    dTdx = []
    dTdy = []
    z = []
    β = []
    dβdx = []
    dβdy = []
    dβdθ = []
    for i in range(N): #note s_prev must have more time steps than state elements
        dTdx.append(jax.jacrev(terrain,argnums = 0)(s_prev[i,0],s_prev[i,1]))
        dTdy.append(jax.jacrev(terrain,argnums = 1)(s_prev[i,0],s_prev[i,1]))
        z.append(terrain(s_prev[i,0],s_prev[i,1]))
        β.append(inclination(s_prev[i,0],s_prev[i,1],s_prev[i,2]))
        dβdx.append(jax.jacrev(inclination,argnums = 0)(s_prev[i,0],s_prev[i,1],s_prev[i,2]))
        dβdy.append(jax.jacrev(inclination,argnums = 1)(s_prev[i,0],s_prev[i,1],s_prev[i,2]))
        dβdθ.append(jax.jacrev(inclination,argnums = 2)(s_prev[i,0],s_prev[i,1],s_prev[i,2]))
    dTdx,dTdy,z = np.array(dTdx),np.array(dTdy),np.array(z)
    dβdx,dβdy,dβdθ = np.array(dβdx),np.array(dβdy),np.array(dβdθ)
    
    cT = []
    cβ = []
    for i in range(N):
        cT.append(z[i] - dTdx[i]*s_prev[i,0] - dTdy[i]*s_prev[i,1])
        cβ.append(β[i] - dβdx[i]*s_prev[i,0] - dβdy[i]*s_prev[i,1] - dβdθ*s_prev[i,2])
    cT = np.array(cT)
    cβ = np.array(cβ)
    z_goal = terrain(s_goal[0],s_goal[1])

    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))
    

    # Construct the convex SCP sub-problem.

    objective = 0.
    constraints = [s_cvx[0]==s0] #enforce initial s
    constraints += [Ad[-1] @ s_cvx[-1] + cd[-1] >= 0]
    for i in range(N):
        objective = objective + α*((s_cvx[i,0] - s_goal[0])**2 + (s_cvx[i,1] - s_goal[1])**2 + (dTdx[i]*s_cvx[i,0] + dTdy[i]*s_cvx[i,1] + cT[i] -z_goal)**2) #stage costs
        objective = objective + (1.-α)*((Ae[i]@s_cvx[i] + Be[i]@u_cvx[i] + ce[i])**2)
        constraints.append(Af[i]@s_cvx[i] + Bf[i]@u_cvx[i] + cf[i] == s_cvx[i+1]) #dynamics
        constraints.append(u_cvx[i,0] <= 50.) #constraint on speed
        constraints.append(u_cvx[i,0] >= -50.) #constraint on speed
        constraints.append(u_cvx[i,1] <= 10.) #constraint on angular speed (turning rate)
        constraints.append(u_cvx[i,1] >=- 10.) #constraint on angular speed (turning rate)
        #constraints.append(dβdx[i]*s_cvx[i,0] + dβdy[i]*s_cvx[i,1] + dβdθ[i]*s_cvx[i,2] + cβ[i] <= 0.3)#constraint on road grade
        #constraints.append(dβdx[i]*s_cvx[i,0] + dβdy[i]*s_cvx[i,1] + dβdθ[i]*s_cvx[i,2] + cβ[i] >= -0.3)#constraint on road grade
        constraints += [Ad[i] @ s_cvx[i] + cd[i] >= 0]

    objective = objective + cvx.quad_form(s_cvx[N][0:2]-s_goal[0:2],P[0:2][:,0:2])  #terminal cost
    

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve(max_iter=1000000)
    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    return s, u, J


def solve_obstacle_avoidance_scp(f, d, s0, s_goal, N, P, Q, R, α, eps, max_iters,
                                 s_init=None, u_init=None,
                                 convergence_error=False):
    """Solve the obstacle avoidance problem via SCP."""
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize trajectory
    if s_init is None or u_init is None:
        s = np.zeros((N + 1, n))
        u = np.zeros((N, m))
        s[0] = s0
        for k in range(N):
            s[k+1] = f(s[k], u[k])
    else:
        s = np.copy(s_init)
        u = np.copy(u_init)

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in range(max_iters):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, P, Q, R,α)
        dJ = np.abs(J[i + 1] - J[i])
        if dJ < eps:
            converged = True
            break
    if not converged and convergence_error:
        raise RuntimeError('SCP did not converge!')
    return s, u


# Define constants
α = 0.4                                 # α = 1 --> only directly consider distance; α = 0 --> only directly consider energy                                   # state dimension
n = 3
m = 2                                   # control dimension
s0 = np.array([1., 1., np.pi/4])        # initial state
s_goal = np.array([9., 9., np.pi/4])    # desired final state
u_final = jnp.array([0.,0.])            # desire zero control at goal state
T = 20                                  # total simulation time
P = 1e2*np.eye(n)                       # terminal state cost matrix
Q = 1e1*np.eye(n)                       # state cost matrix
R = np.eye(m)                           # control cost matrix
eps = 1e-3                              # SCP convergence tolerance

A, B = jax.jacfwd(dynamics, (0,1))(s_goal,u_final)
A, B = jnp.array(A), jnp.array(B)
P = jnp.linalg.solve(Q+A.T@P@A, P) # Get P from the Lyapnov equation


# Set obstacle center points and radii
# centers = np.array([
#     [3.6,5.1],
# ])

# Used for actual project
centers = np.array([
    [4,6]
])
radii = np.array([2])

# centers = np.array([
#     [1,8]
# ])
# radii = np.array([0.5])



N = 20 # MPC horizon
N_scp = 10    # maximum number of SCP iterations

f = dynamics
d = partial(signed_distances, centers=centers, radii=radii)
s_mpc = np.zeros((T, N + 1, n))
u_mpc = np.zeros((T, N, m))
s = np.copy(s0)
total_time = time()
total_control_cost = 0.
s_init = None
u_init = None
for t in tqdm(range(T)):
    # Solve the MPC problem at time `t`
    
    s_mpc[t], u_mpc[t] = solve_obstacle_avoidance_scp(f,d,s,s_goal,N,P,Q,R,α,eps,N_scp)

    # Push the state `s` forward in time with a closed-loop MPC input
    s = f(s,u_mpc[t,0])

    # Accumulate the actual control cost
    total_control_cost += u_mpc[t, 0].T @ R @ u_mpc[t, 0]

    # Use this solution to warm-start the next iteration
    u_init = np.concatenate([u_mpc[t, 1:], u_mpc[t, -1:]])
    s_init = np.concatenate([
        s_mpc[t, 1:],
        f(s_mpc[t, -1], u_mpc[t, -1]).reshape([1, -1])
    ])
total_time = time() - total_time
print('Total elapsed time:', total_time, 'seconds')
print('Total control cost:', total_control_cost)

fig, ax = plt.subplots(1, 2, dpi=150, figsize=(15, 5))
fig.suptitle('$N = {}$, '.format(N)+ r'$N_\mathrm{SCP} = ' + '{}$, '.format(N_scp) + 'alpha = ' + str(α))

# # Uncomment to display obstacles
# for pc, rc in zip(centers, radii):
#     ax[0].add_patch(
#         plt.Circle((pc[0], pc[1]), rc, color='r', alpha=0.3)
#     )

for t in range(T):
    ax[0].plot(s_mpc[t, :, 0], s_mpc[t, :, 1], '--*', color='k')
ax[0].plot(s_mpc[:, 0, 0], s_mpc[:, 0, 1], '-o')
ax[0].set_xlabel(r'$x(t)$')
ax[0].set_ylabel(r'$y(t)$')
ax[0].axis('equal')

ax[1].plot(u_mpc[:, 0, 0], '-o', label=r'$u_1(t)$')
ax[1].plot(u_mpc[:, 0, 1], '-o', label=r'$u_2(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$u(t)$')
ax[1].legend()


print("Final State")
print("x: " + str(s_mpc[-1, 0, 0]) + ", y: " + str(s_mpc[-1, 0, 1]))

suffix = '_N={}_Nscp={}'.format(N, N_scp)
# plt.savefig('soln_obstacle_avoidance' + suffix + '.png', bbox_inches='tight')
plt.savefig('update_proj_results_N=' + str(N) + '_NSCP=' +str(N_scp) + '_alpha=' + str(α) + '_T=' + str(T) + '.png', bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0,9.0,0.1)
y = np.arange(0.0,9.0,0.1)
elev = np.zeros((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        elev[i,j] = terrain(x[i],y[j])
elev = np.flipud(elev.T)
plt.imshow(elev,interpolation = 'none')
plt.colorbar()

plt.figure(2)
energy_vec = []
for t in range(s_mpc.shape[0]):
    energy_vec.append(energy_for_plotting(s_mpc[t,t,:], u_mpc[t,t,:]))
# print(energy_vec)
print(np.mean(np.array(energy_vec)))
np.save("energy_vec_alpha_" + str(α), np.array(energy_vec))
plt.plot(energy_vec)
plt.show()
