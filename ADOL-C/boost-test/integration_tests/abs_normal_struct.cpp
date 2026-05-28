#include "../const.h"
#include "adolc/tape_interface.h"
#include <adolc/adolc.h>
#include <adolc/drivers/psdrivers.h>
#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <vector>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(AbsNormalFormStructTest)

namespace {

template <typename ActualContainer, typename ExpectedContainer>
void checkCloseContainer(const ActualContainer &actual,
                         const ExpectedContainer &expected,
                         const char *containerName) {
  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_TEST_CONTEXT(containerName << "[" << i << "]") {
      BOOST_TEST(actual[i] == expected[i], tt::tolerance(tol));
    }
  }
}

template <size_t Rows, size_t Cols>
void checkCloseMatrix(
    double *const *actual, const std::array<std::array<double, Cols>, Rows> &expected,
    const char *matrixName) {
  for (size_t row = 0; row < Rows; ++row) {
    for (size_t col = 0; col < Cols; ++col) {
      BOOST_TEST_CONTEXT(matrixName << "(" << row << "," << col << ")") {
        BOOST_TEST(actual[row][col] == expected[row][col], tt::tolerance(tol));
      }
    }
  }
}

struct SimpleAbsProblem {
  static constexpr size_t dimIn = 2;
  static constexpr size_t dimOut = 1;

  short tapeId{-1};
  std::array<double, dimIn> x = {1.0, -2.0};
  std::array<double, dimOut> y{};

  size_t numSwitches{0};

  SimpleAbsProblem() : tapeId(createNewTape()) {}
};

void taping(SimpleAbsProblem &problem) {
  setCurrentTape(problem.tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(problem.tapeId);

  std::vector<adouble> ax(SimpleAbsProblem::dimIn);
  std::vector<adouble> ay(SimpleAbsProblem::dimOut);

  for (int i = 0; i < SimpleAbsProblem::dimIn; i++) {
    ax[i] <<= problem.x[i];
  }

  const auto z1 = fabs(ax[0]);
  const auto z2 = fabs(ax[1]);
  ay[0] = ax[0] + ax[1] - z1 - z2;

  ay[0] >>= problem.y[0];
  trace_off();

  problem.numSwitches = get_num_switches(problem.tapeId);
  BOOST_TEST(problem.numSwitches == 2);
}

} // namespace

BOOST_AUTO_TEST_CASE(AbsNormalForm_Struct) {
  SimpleAbsProblem problem{};
  taping(problem);

  // Use the new struct-based API
  ADOLC::DenseAbsNormalForm anf(SimpleAbsProblem::dimIn, SimpleAbsProblem::dimOut,
                                problem.numSwitches);

  const int rc = ADOLC::abs_normal(problem.tapeId, problem.x.data(), anf);

  BOOST_TEST(rc == 0);
  checkCloseContainer(anf.y, std::array<double, 1>{-4.0}, "y");
  checkCloseContainer(anf.z, std::array<double, 2>{1.0, -2.0}, "z");
  checkCloseContainer(anf.cz, std::array<double, 2>{1.0, -2.0}, "cz");
  checkCloseContainer(anf.cy, std::array<double, 1>{-1.0}, "cy");

  checkCloseMatrix(
      anf.L.data(), std::array<std::array<double, 2>, 2>{{{0.0, 0.0}, {0.0, 0.0}}},
      "L");
  checkCloseMatrix(
      anf.Z.data(), std::array<std::array<double, 2>, 2>{{{1.0, 0.0}, {0.0, 1.0}}},
      "Z");
  checkCloseMatrix(anf.Y.data(), std::array<std::array<double, 2>, 1>{{{1.0, 1.0}}},
                   "Y");
  checkCloseMatrix(anf.J.data(),
                   std::array<std::array<double, 2>, 1>{{{-1.0, -1.0}}}, "J");
}

BOOST_AUTO_TEST_CASE(AbsNormalForm_Struct_FromTape) {
  SimpleAbsProblem problem{};
  taping(problem);

  // Automatically extract dimensions from the tape
  auto anf = ADOLC::DenseAbsNormalForm::fromTape(problem.tapeId);

  BOOST_TEST(anf.n == SimpleAbsProblem::dimIn);
  BOOST_TEST(anf.m == SimpleAbsProblem::dimOut);
  BOOST_TEST(anf.s == problem.numSwitches);

  const int rc = ADOLC::abs_normal(problem.tapeId, problem.x.data(), anf);

  BOOST_TEST(rc == 0);
  checkCloseContainer(anf.y, std::array<double, 1>{-4.0}, "y");
  checkCloseContainer(anf.z, std::array<double, 2>{1.0, -2.0}, "z");
}

BOOST_AUTO_TEST_SUITE_END()
