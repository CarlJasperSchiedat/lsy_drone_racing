from lsy_drone_racing.control.attitude_mpc import export_quadrotor_ode_model
from acados_template import AcadosOcp, AcadosOcpSolver
import scipy.linalg
import numpy as np
import os

# Setze ACADOS Pfad korrekt
os.environ["ACADOS_SOURCE_DIR"] = "C:/Users/conda/repos/lsy_drone_racing/acados"

Tf = 1.5
N = 30

model = export_quadrotor_ode_model()

ocp = AcadosOcp()
ocp.model = model
ocp.dims.N = N
ocp.solver_options.tf = Tf
ocp.code_export_directory = "c_generated_code"
ocp.json_file = "my_mpc_controller.json"

nx = model.x.rows()
nu = model.u.rows()
ny = 15  # angepasst, siehe vorherige Korrektur
ny_e = nx

Q = np.diag([10.0]*3 + [0.01]*3 + [0.1]*3 + [0.01]*2)  # 11 Elemente
R = np.diag([0.01]*4)  # 4 Elemente
ocp.cost.W = scipy.linalg.block_diag(Q, R)

Vx = np.zeros((ny, nx))
Vx[:Q.shape[0], :] = np.eye(nx)[:Q.shape[0], :]
Vu = np.zeros((ny, nu))
Vu[Q.shape[0]:, :] = np.eye(nu)

ocp.cost.cost_type = "LINEAR_LS"
ocp.cost.cost_type_e = "LINEAR_LS"
ocp.cost.W = scipy.linalg.block_diag(Q, R)
ocp.cost.Vx = Vx
ocp.cost.Vu = Vu

ocp.cost.W_e = np.eye(nx)                # Terminal weight
ocp.cost.Vx_e = np.eye(nx)               # Full-state tracking
ocp.cost.yref = np.zeros((ny,))
ocp.cost.yref_e = np.zeros((nx,))
ocp.constraints.x0 = np.zeros(nx)

ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.nlp_solver_type = "SQP"

print("🔧 Generiere ACADOS Code...")
AcadosOcpSolver.generate(ocp, json_file=ocp.json_file)
print("✅ Fertig. DLL sollte in c_generated_code/ liegen.")
