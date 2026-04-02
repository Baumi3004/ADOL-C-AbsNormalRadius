#include <adolc/adolc.h>
#include <adolc/drivers/psdrivers.h>
#include <iostream>
#include <string_view>
#include <vector>
#include <cmath>

struct ADProblem {
  static constexpr size_t dimIn = 2;
  static constexpr size_t dimOut = 1;

  short tapeId{-1};
  std::vector<double> x = {0.4, -0.1};
  std::vector<double> y{0.0};
  double radius = 0.5;

  size_t numSwitches{0};

  ADProblem() : tapeId(createNewTape()) {}
};

void taping(ADProblem &problem) {
  currentTape().enableMinMaxUsingAbs();
  trace_on(problem.tapeId);

  std::vector<adouble> ax(ADProblem::dimIn);
  std::vector<adouble> ay(ADProblem::dimOut);

  for (int i = 0; i < ADProblem::dimIn; i++)
    ax[i] <<= problem.x[i];

  // Example function with absolute values
  ay[0] = ax[0] + ax[1] - fabs(ax[0]) - fabs(ax[1]);

  ay[0] >>= problem.y[0];
  trace_off();

  problem.numSwitches = get_num_switches(problem.tapeId);
  std::cout << "Total switches (s_full) = " << problem.numSwitches << "\n";
}

void printMatrix(std::string_view description, const std::vector<double *> &matrix,
                 size_t dimx, size_t dimy) {

  std::cout << "--- " << description << " ---\n";
  for (size_t i = 0; i < dimx; ++i) {
    for (size_t j = 0; j < dimy; ++j) {
      std::cout << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}

void computeReducedAbsNormal(ADProblem &problem) {
  absLinearFormReduced alfr;
  
  // Define the callback to determine which switches are "reduced" (active)
  auto compute_reduced = [&problem](const std::vector<double> &x, 
                                   const std::vector<double> &z_full, 
                                   std::vector<bool> &is_switch) {
    std::cout << "Evaluating reduced switch set with radius = " << problem.radius << "\n";
    for (size_t i = 0; i < z_full.size(); ++i) {
      is_switch[i] = (std::abs(z_full[i]) < problem.radius);
      std::cout << "  Switch " << i << ": |z| = " << std::abs(z_full[i]) 
                << " < " << problem.radius << " ? " << (is_switch[i] ? "YES" : "NO") << "\n";
    }
    return 0; // success
  };

  int rc = abs_normal_reduced(problem.tapeId, problem.x, alfr, compute_reduced);

  std::cout << "abs_normal_reduced return code = " << rc << "\n";
  std::cout << "Reduced switches (s_red) = " << alfr.s << "\n";

  if (alfr.s > 0) {
    printMatrix("L_red (s_red x s_red)", alfr.L, alfr.s, alfr.s);
    printMatrix("Z_red (s_red x n)", alfr.Z, alfr.s, alfr.n);
    printMatrix("B_red (m x s_red)", alfr.B, alfr.m, alfr.s);
  } else {
    std::cout << "No switches are reduced (active).\n";
  }
  printMatrix("A_red (m x n)", alfr.A, alfr.m, alfr.n);
}

int main() {
  ADProblem problem{};
  
  std::cout << "Taping...\n";
  taping(problem);
  
  std::cout << "\nComputing Reduced ABS-Normal Form...\n";
  computeReducedAbsNormal(problem);

  return 0;
}
