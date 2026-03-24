/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/psdrivers.cpp
 Revision: $Id$
 Contents: Easy to use drivers for piecewise smooth functions
           (with C and C++ callable interfaces including Fortran
            callable versions).

 Copyright (c) Andrea Walther, Sabrina Fiege

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduct ion, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/drivers/psdrivers.h>
#include <adolc/dvlparms.h>
#include <adolc/interfaces.h>
#include <adolc/internal/common.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <math.h>
#include <vector>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                 DRIVERS FOR PS FUNCTIONS */

/*--------------------------------------------------------------------------*/
/*                                                          abs-normal form */

int abs_normal_struct(short tag, const std::vector<double> &x,
                      absLinearForm &alf) {
  // get indep, deps, and num_switches from the tape
  ValueTape &tape = findTape(tag);
  tape.init_sweep<ValueTape::Forward>();
  size_t m = tape.tapestats(TapeInfos::NUM_DEPENDENTS);
  size_t n = tape.tapestats(TapeInfos::NUM_INDEPENDENTS);
  size_t s = tape.tapestats(TapeInfos::NUM_SWITCHES);
  tape.end_sweep();
  // reallocate memory if struct dims didnt match
  if (alf.s != s || alf.n != n || alf.m != m) {
    alf.y.resize(m);
    alf.z.resize(s);
    alf.cz.resize(s);
    alf.cy.resize(m);
    alf.A_mem.resize(m * n);
    alf.B_mem.resize(m * s);
    alf.Z_mem.resize(s * n);
    alf.L_mem.resize(s * s);
    alf.A.resize(m);
    alf.B.resize(m);
    alf.Z.resize(s);
    alf.L.resize(s);
    for (size_t i = 0; i < s; i++) {
      alf.Z[i] = &alf.Z_mem.data()[i * n];
      alf.L[i] = &alf.L_mem.data()[i * s];
    }
    for (size_t i = 0; i < m; i++) {
      alf.A[i] = &alf.A_mem.data()[i * n];
      alf.B[i] = &alf.B_mem.data()[i * s];
    }
  }
  alf.n = n;
  alf.m = m;
  alf.s = s;

  return abs_normal(tag, static_cast<int>(alf.m), static_cast<int>(alf.n),
                    static_cast<int>(alf.s), x.data(), alf.y.data(),
                    alf.z.data(), alf.cz.data(), alf.cy.data(), alf.A.data(),
                    alf.B.data(), alf.Z.data(), alf.L.data());
}

int abs_normal(short tag,       /* tape identifier */
               int m,           /* number od dependents   */
               int n,           /* number of independents */
               int swchk,       /* number of switches (check) */
               const double *x, /* base point */
               double *y,       /* function value */
               double *z,       /* switching variables */
               double *cz,      /* first constant */
               double *cy,      /* second constant */
               double **Y,      /* m times n */
               double **J,      /* m times s */
               double **Z,      /* s times n */
               double **L)      /* s times s (lowtri) */
{

  const size_t s = get_num_switches(tag);
  /* This check is required because the user is probably allocating his
   * arrays sigma, cz, Z, L, Y, J according to swchk */
  if (s != to_size_t(swchk))
    ADOLCError::fail(
        ADOLCError::ErrorType::SWITCHES_MISMATCH, CURRENT_LOCATION,
        ADOLCError::FailInfo{.info1 = tag, .info3 = swchk, .info6 = s});

  std::vector<double> res(n + s);

  zos_pl_forward(tag, m, n, 1, x, y, z);
  // loop trough first s rows for switched
  for (size_t i = 0; i < s; i++) {
    fos_pl_reverse(tag, m, n, static_cast<int>(s), static_cast<int>(i),
                   res.data());
    for (int j = 0; j < n; j++) {
      Z[i][j] = res[j];
    }
    // cz = z - L|z|
    cz[i] = z[i];
    for (size_t j = 0; j < s; j++) {
      /* L[i][i] .. L[i][s] are theoretically zero,
       *  we probably don't need to copy them */
      L[i][j] = res[j + n];
      if (j < i) {
        cz[i] = cz[i] - L[i][j] * fabs(z[j]);
      }
    }
  }
  // loop through deps with rows s+i
  for (int i = 0; i < m; i++) {
    fos_pl_reverse(tag, m, n, static_cast<int>(s), static_cast<int>(s + i),
                   res.data());
    for (int j = 0; j < n; j++) {
      Y[i][j] = res[j];
    }
    // cy = y - L|z|
    cy[i] = y[i];
    for (size_t j = 0; j < s; j++) {
      J[i][j] = res[j + n];
      cy[i] = cy[i] - J[i][j] * fabs(z[j]);
    }
  }
  return 0;
}

/*--------------------------------------------------------------------------*/
int abs_normal_almost_active(
    short tag, const std::vector<double> &x, absLinearFormAlmostActive &alfaa,
    std::function<int(const std::vector<double> &x,
                      const std::vector<double> &z_full,
                      std::vector<bool> &is_almost_active)>
        compute_almost_active) {

  // get indep, deps, and num_switches from the tape
  ValueTape &tape = findTape(tag);
  tape.init_sweep<ValueTape::Forward>();
  size_t m = tape.tapestats(TapeInfos::NUM_DEPENDENTS);
  size_t n = tape.tapestats(TapeInfos::NUM_INDEPENDENTS);
  size_t s_full = tape.tapestats(TapeInfos::NUM_SWITCHES);
  tape.end_sweep();

  // reallocate memory required for forward if struct dims didnt match
  if (alfaa.s_full != s_full || alfaa.n != n || alfaa.m != m) {
    alfaa.y.resize(m);
    alfaa.z_full.resize(s_full);
    alfaa.cy.resize(m);
    alfaa.A_mem.resize(m * n);
    alfaa.A.resize(m);
    for (size_t i = 0; i < m; i++) {
      alfaa.A[i] = &alfaa.A_mem.data()[i * n];
    }
    alfaa.is_almost_active.resize(s_full);
    alfaa.s_full = s_full;
    alfaa.m = m;
    alfaa.n = n;
    alfaa.s = -1;
  }

  zos_pl_forward(tag, static_cast<int>(m), static_cast<int>(n), 1, x.data(),
                 alfaa.y.data(), alfaa.z_full.data());
  // CALLBACK
  compute_almost_active(x, alfaa.z_full, alfaa.is_almost_active);

  size_t s = std::count(alfaa.is_almost_active.begin(),
                        alfaa.is_almost_active.end(), true);
  //  change sizes for variable matrices
  if (alfaa.s != s) {
    alfaa.z.resize(s);
    alfaa.cz.resize(s);
    alfaa.B_mem.resize(m * s);
    alfaa.Z_mem.resize(s * n);
    alfaa.L_mem.resize(s * s);
    alfaa.B.resize(m);
    alfaa.Z.resize(s);
    alfaa.L.resize(s);
    for (size_t i = 0; i < s; i++) {
      alfaa.Z[i] = &alfaa.Z_mem.data()[i * n];
      alfaa.L[i] = &alfaa.L_mem.data()[i * s];
    }
    for (size_t i = 0; i < m; i++) {
      alfaa.B[i] = &alfaa.B_mem.data()[i * s];
    }
    alfaa.s = s;
  }

  std::vector<double> res(n + s_full);

  // compute L, Z matrices for almostactive switches
  int row = 0;
  for (size_t i = 0; i < s_full; i++) {
    if (alfaa.is_almost_active[i]) {
      fos_pl_reverse_almost_active(
          tag, static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(s_full), static_cast<int>(i), res.data(),
          alfaa.is_almost_active);
      alfaa.z[row] = alfaa.z_full[i];
      for (size_t j = 0; j < n; j++) {
        alfaa.Z[row][j] = res[j];
      }
      // cz = z - L|z|
      alfaa.cz[row] = alfaa.z_full[i];
      int col = 0;
      for (size_t j = 0; j < s_full; j++) {
        /* L[i][i] .. L[i][s_full] are theoretically zero,
         *  we probably don't need to copy them */
        if (alfaa.is_almost_active[j]) {
          alfaa.L[row][col] = res[j + n];
          if (j < i) {
            alfaa.cz[row] =
                alfaa.cz[row] - alfaa.L[row][col] * fabs(alfaa.z_full[j]);
          }
          col++;
        }
      }
      row++;
    }
  }
  // compute A, B with argument s+i for row
  for (size_t i = 0; i < m; i++) {
    fos_pl_reverse_almost_active(
        tag, static_cast<int>(m), static_cast<int>(n), static_cast<int>(s_full),
        static_cast<int>(s_full + i), res.data(), alfaa.is_almost_active);
    for (size_t j = 0; j < n; j++) {
      alfaa.A[i][j] = res[j];
    }
    // cy = y - L|z|
    alfaa.cy[i] = alfaa.y[i];
    int col = 0;
    for (size_t j = 0; j < s_full; j++) {
      if (alfaa.is_almost_active[j]) {
        alfaa.B[i][col] = res[j + n];
        alfaa.cy[i] = alfaa.cy[i] - alfaa.B[i][col] * fabs(alfaa.z_full[j]);
        col++;
      }
    }
  }
  return 0;
}

/*--------------------------------------------------------------------------*/
/*                                              directional_active_gradient */
/*                                                                          */
int directional_active_gradient(short tag,       /* trace identifier */
                                int n,           /* number of independents */
                                const double *x, /* value of independents */
                                const double *d, /* direction */
                                double *g,     /* directional active gradient */
                                short *sigma_g /* sigma of g */
) {
  int max_dk, keep;
  double max_entry, y, by;
  double *z;
  double **E, **grad, **gradu;

  keep = 1;
  by = 1;

  const size_t s = get_num_switches(tag);

  z = myalloc1(s);

  grad = (double **)myalloc2(1, n);
  gradu = (double **)myalloc2(s, n);
  E = (double **)myalloc2(n, n);

  max_dk = 0;
  max_entry = -1;
  for (int i = 0; i < n; i++) {
    E[i][0] = d[i];
    if (max_entry < fabs(d[i])) {
      max_dk = i;
      max_entry = fabs(d[i]);
    }
  }

  int k = 1;
  bool done = 0;
  int j = 0;

  while ((k < 6) && (done == 0)) {
    fov_pl_forward(tag, 1, n, k, x, E, &y, grad, z, gradu, sigma_g);

    size_t sum = 0;
    for (size_t i = 0; i < s; i++) {
      sum += abs(sigma_g[i]);
    }

    if (sum == s) {
      zos_pl_forward(tag, 1, n, keep, x, &y, z);
      // the cast is necessary since the type signature uses "int". Its now
      // explicit.
      fos_pl_sig_reverse(tag, 1, n, static_cast<int>(s), sigma_g, &by, g);
      done = 1;
    } else {
      if (j == max_dk)
        j++;
      E[j][k] = 1;
      j++;
      k++;
    }
  }

  myfree1(z);
  myfree2(E);
  myfree2(grad);
  myfree2(gradu);

  if (done == 0)
    ADOLCError::fail(ADOLCError::ErrorType::DIRGRAD_NOT_ENOUGH_DIRS,
                     CURRENT_LOCATION);
  return 0;
}

END_C_DECLS
