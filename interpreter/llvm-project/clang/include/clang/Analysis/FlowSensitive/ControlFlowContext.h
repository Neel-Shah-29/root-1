//===-- ControlFlowContext.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a ControlFlowContext class that is used by dataflow
//  analyses that run over Control-Flow Graphs (CFGs).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_CONTROLFLOWCONTEXT_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_CONTROLFLOWCONTEXT_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <utility>

namespace clang {
namespace dataflow {

/// Holds CFG and other derived context that is needed to perform dataflow
/// analysis.
class ControlFlowContext {
public:
  /// Builds a ControlFlowContext from an AST node. `D` is the function in which
  /// `S` resides and must not be null.
  static llvm::Expected<ControlFlowContext> build(const Decl *D, Stmt &S,
                                                  ASTContext &C);

  /// Returns the `Decl` containing the statement used to construct the CFG, if
  /// available.
  const Decl *getDecl() const { return ContainingDecl; }

  /// Returns the CFG that is stored in this context.
  const CFG &getCFG() const { return *Cfg; }

  /// Returns a mapping from statements to basic blocks that contain them.
  const llvm::DenseMap<const Stmt *, const CFGBlock *> &getStmtToBlock() const {
    return StmtToBlock;
  }

private:
  // FIXME: Once the deprecated `build` method is removed, mark `D` as "must not
  // be null" and add an assertion.
  ControlFlowContext(const Decl *D, std::unique_ptr<CFG> Cfg,
                     llvm::DenseMap<const Stmt *, const CFGBlock *> StmtToBlock)
      : ContainingDecl(D), Cfg(std::move(Cfg)),
        StmtToBlock(std::move(StmtToBlock)) {}

  /// The `Decl` containing the statement used to construct the CFG.
  const Decl *ContainingDecl;
  std::unique_ptr<CFG> Cfg;
  llvm::DenseMap<const Stmt *, const CFGBlock *> StmtToBlock;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_CONTROLFLOWCONTEXT_H
