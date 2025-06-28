from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat
import numpy as np


import os

# Windows-kompatible Umgebungsvariablen setzen
os.environ["CC"] = "gcc"
os.environ["LD"] = "gcc"
os.environ["RM"] = "del"



def export_test_model():
    model = AcadosModel()
    model.name = "simple_test"

    # States and controls
    x = MX.sym("x")
    u = MX.sym("u")

    # Dynamics: ẋ = -x + u
    x_dot = -x + u

    model.x = vertcat(x)
    model.u = vertcat(u)
    model.f_expl_expr = vertcat(x_dot)
    model.f_impl_expr = model.f_expl_expr - MX.sym("xdot", 1)

    return model

def test_acados_installation():
    # Create model and ocp object
    model = export_test_model()
    ocp = AcadosOcp()
    ocp.model = model

    # Dimensions
    ocp.dims.N = 20

    # Cost: least-squares tracking to 0
    Q = np.array([[1.0]])
    R = np.array([[0.1]])
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx = np.array([[1.0], [0.0]])  # ny=2, nx=1
    ocp.cost.Vu = np.array([[0.0], [1.0]])  # ny=2, nu=1
    ocp.cost.W = np.eye(2)  # (2,2)
    ocp.cost.yref = np.zeros((2,))

    ocp.cost.Vx_e = np.array([[1.0]])  # ny_e=1, nx=1
    ocp.cost.W_e = np.array([[1.0]])
    ocp.cost.yref_e = np.zeros((1,))


    # Constraints
    ocp.constraints.x0 = np.array([1.0])

    # Solver settings
    ocp.solver_options.tf = 2.0
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver_warm_start = 1



    # Constraints
    ocp.constraints.x0 = np.array([1.0])
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.lbx = np.array([1.0])
    ocp.constraints.ubx = np.array([1.0])

    # Create solver
    ocp_solver = AcadosOcpSolver(ocp, json_file="test_ocp.json")

    # Solve OCP
    for i in range(ocp.dims.N):
        ocp_solver.set(i, "yref", np.zeros((2,)))
    ocp_solver.set(ocp.dims.N, "yref", np.zeros((1,)))



    status = ocp_solver.solve()
    if status != 0:
        print(f"❌ acados solver failed with status {status}")
    else:
        print("✅ acados solver successfully solved the OCP")

if __name__ == "__main__":
    test_acados_installation()
