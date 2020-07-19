from fenics import * 
from mshr import *
import matplotlib.pyplot as plt 
import numpy as np 
import time
import matplotlib.pyplot as plt
import cupy
import cupyx
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
mempool = cupy.get_default_memory_pool()
with cupy.cuda.Device(0):
    mempool.set_limit(size=1.2*1024**3)
parameters['linear_algebra_backend'] = 'Eigen'
def tran2SparseMatrix(A):
    row, col, val = as_backend_type(A).data()
    return sps.csr_matrix((val, col, row))
T = 2.0            # final time
num_steps = 5000   # number of time steps
dt = T / num_steps # time step size
mu = 0.001         # dynamic viscosity
rho = 1            # density

# Create mesh
channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = generate_mesh(domain, 60)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
b1 = assemble(L1)
b2 = assemble(L2)
b3 = assemble(L3)

print("Grid Points:",np.size(b1[:]))

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

#Converting to Sparse Matrix
A1 = tran2SparseMatrix(A1)
A2 = tran2SparseMatrix(A2)
A3 = tran2SparseMatrix(A3)

# Create XDMF files for visualization output
xdmffile_u = XDMFFile('navier_GPU/velocity.xdmf')
xdmffile_p = XDMFFile('navier_GPU/pressure.xdmf')


# Time-stepping
t = 0
start = time.time()
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    b1 = b1[:]
    As1 = cupyx.scipy.sparse.csr_matrix(A1)
    bs1 = cupy.array(b1)
    u_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As1, bs1)[:1][0])

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    b2 = b2[:]
    As2 = cupyx.scipy.sparse.csr_matrix(A2)
    bs2 = cupy.array(b2)
    p_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As2, bs2)[:1][0])

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    b3 = b3[:]
    As3 = cupyx.scipy.sparse.csr_matrix(A3)
    bs3 = cupy.array(b3)
    u_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As3, bs3)[:1][0])

    # Plot solution
    #plot(u_, title='Velocity')
    #plt.show()
    #plot(p_, title='Pressure')
    #plt.show()

    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    print('Time:', t)
    print('u max:', u_.vector().max())
    print('p max:', p_.vector().max())
    
end = time.time()
print("lsqr_GPU(s)",end - start)
    
