#include <crocoddyl/core/states/state-euclidean.hpp>
#include <crocoddyl/core/actions/unicycle.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>
#include <crocoddyl/core/solvers/ddp.hpp>

#include <ctime>


int main() {
  bool CALLBACKS = false;
  unsigned int N = 200; // number of nodes
  unsigned int T = 5e3; // number of trials
  unsigned int MAXITER = 1;
  using namespace crocoddyl;

  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<ActionModelAbstract*> runningModels;
  ActionModelAbstract* terminalModel;
  x0 = Eigen::Vector3d(1., 0., 0.);
  StateVector state(3);

  // Creating the action models and warm point for the unicycle system
  for (unsigned int i = 0; i < N; ++i) {
    ActionModelAbstract* model_i = new ActionModelUnicycle(&state);
    runningModels.push_back(model_i);
    xs.push_back(x0);
    us.push_back(Eigen::Vector2d::Zero());
  }
  xs.push_back(x0);
  terminalModel = new ActionModelUnicycle(&state);

  // Formulating the optimal control problem
  ShootingProblem problem(x0, runningModels, terminalModel);
  SolverDDP ddp(problem);
  if (CALLBACKS) {
    std::vector<CallbackAbstract*> cbs;
    cbs.push_back(new CallbackDDPVerbose());
    ddp.setCallbacks(cbs);
  }

  // Solving the optimal control problem
  std::clock_t c_start, c_end;
  std::vector<double> duration;
  for (unsigned int i = 0; i < T; ++i) {
    c_start = std::clock();
    ddp.solve(xs, us, MAXITER);
    c_end = std::clock();
    duration.push_back(1e3 * (double) (c_end - c_start) / CLOCKS_PER_SEC);
  }

  double avrg_duration=0., min_duration=std::numeric_limits<double>::max(), max_duration=0.;
  for (unsigned int i = 0; i < T; ++i) {
    const double& dt = duration[i];
    avrg_duration += dt;
    if (dt < min_duration) {
      min_duration = dt;
    }
    if (dt > max_duration) {
      max_duration = dt;
    }
  }
  avrg_duration /= T;
  std::cout << "CPU time [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")" << std::endl;
}
