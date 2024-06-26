// @(#)root/mathmore:$Id$
// Authors: L. Moneta, 08/2007

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU General Public License        *
  * as published by the Free Software Foundation; either version 2     *
  * of the License, or (at your option) any later version.             *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * General Public License for more details.                           *
  *                                                                    *
  * You should have received a copy of the GNU General Public License  *
  * along with this library (see file COPYING); if not, write          *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact the author.             *
  *                                                                    *
  **********************************************************************/

// Header file for class GSLMonteFunctionWrapper
//
// Created by: moneta  at Sat Nov 13 14:54:41 2004
//
// Last update: Sat Nov 13 14:54:41 2004
//
#ifndef ROOT_Math_GSLMonteFunctionWrapper
#define ROOT_Math_GSLMonteFunctionWrapper

#include "gsl/gsl_monte.h"
#include "gsl/gsl_multimin.h"

#include "GSLMonteFunctionAdapter.h"


#include <cassert>

namespace ROOT {
namespace Math {



   typedef double ( * GSLMonteFuncPointer ) ( double *, size_t, void *);


/**
   wrapper to a multi-dim function withtout  derivatives for Monte Carlo multi-dimensional
   integration algorithm

   @ingroup MCIntegration
*/

class GSLMonteFunctionWrapper {

public:

  GSLMonteFunctionWrapper()
   {
      fFunc.f = nullptr;
      fFunc.dim = 0;
      fFunc.params = nullptr;
   }

    void SetFuncPointer( GSLMonteFuncPointer f) { fFunc.f = f; }
    void SetDim  ( unsigned int n ) { fFunc.dim = n; }
    void SetParams ( void * p) { fFunc.params = p; }

    /// Fill gsl function structure from a C++ Function class
    template<class FuncType>
    void SetFunction(const FuncType &f) {
       const void * p = &f;
       assert (p != nullptr);
       SetFuncPointer(&GSLMonteFunctionAdapter<FuncType >::F);
       SetDim( f.NDim() );
       SetParams(const_cast<void *>(p));
    }

   gsl_monte_function * GetFunc() { return &fFunc; }

    /// evaluate the function and derivatives
    double operator() (const double * x) {  return GSL_MONTE_FN_EVAL(&fFunc, const_cast<double *>(x) ); }


  private:
    gsl_monte_function fFunc;

  };





} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_GSLMonteFunctionWrapper */
