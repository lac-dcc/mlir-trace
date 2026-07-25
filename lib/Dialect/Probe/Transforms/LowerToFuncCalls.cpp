#include "Dialect/Probe/IR/Probe.h"
#include "Dialect/Probe/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::probe {
#define GEN_PASS_DEF_LOWERTOFUNCCALLSPASS
#include "Dialect/Probe/Transforms/Passes.h.inc"

namespace {
class LowerToFuncCallsPass
    : public impl::LowerToFuncCallsPassBase<LowerToFuncCallsPass> {
private:
  /// Memory space for creating memref types
  static constexpr unsigned kDefaultMemorySpace = 0;

  /// Local registry of declared probe functions
  llvm::StringSet<> probeFuncRegistry;

  /// Get short string representation for \p elemTy
  static inline std::string getTypeAsStr(Type elemTy) {
    bool isFloat = elemTy.isFloat();
    bool isSignlessInt = elemTy.isSignlessInteger();
    bool isSignedInt = elemTy.isSignedInteger();
    // Treat signless as unsigned.
    std::string typePrefix =
        isFloat ? "F" : (isSignlessInt ? "I" : (isSignedInt ? "S" : "U"));

    switch (elemTy.getIntOrFloatBitWidth()) {
    case 8:
      return typePrefix + "8";
    case 16:
      return typePrefix + "16";
    case 32:
      return typePrefix + "32";
    case 64:
      return typePrefix + "64";
    default:
      llvm_unreachable("Unsupported element type bit width");
    }
  }

  /// Get observe function for memrefs of \p elemTy
  std::string getObserveFuncName(Type elemTy) const {
    return observeFuncPrefix + getTypeAsStr(elemTy);
  }

  /// Get function type for observe function for memref os \p elemTy
  static inline FunctionType getObserveFuncType(MLIRContext *ctx, Type elemTy) {
    auto tensorType = UnrankedMemRefType::get(elemTy, kDefaultMemorySpace);
    auto i32Type = IntegerType::get(ctx, 32);
    return FunctionType::get(ctx, {tensorType, i32Type, i32Type}, {});
  }

  /// Declare a probe function with name \p funcName, if not already present
  /// in \p moduleOp
  void lookupOrCreateProbeFunc(IRRewriter &rewriter, ModuleOp moduleOp,
                               llvm::StringRef funcName, FunctionType funcTy) {
    if (probeFuncRegistry.contains(funcName))
      return;

    // Lookup in the module's SymbolTable if not found in local registry
    if (moduleOp.lookupSymbol(funcName))
      return;

    probeFuncRegistry.insert(funcName);

    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto probeFunc =
        func::FuncOp::create(rewriter, moduleOp.getLoc(), funcName, funcTy);

    // Mark as private declaration, since it will be linked externally
    probeFunc.setPrivate();
  }

  /// Lower probe.observe ops in \p funcOp
  void lowerObserveOps(ModuleOp moduleOp, func::FuncOp funcOp) {
    std::vector<ObserveOp> observeOps;
    auto result =
        funcOp.walk(
            [&](ObserveOp op) {
              if (!llvm::isa<MemRefType>(op.getInput().getType())) {
                op.emitError()
                    << "Expected ranked memref type for observe function call.";
                return WalkResult::interrupt();
              }
              observeOps.push_back(op);
              return WalkResult::advance();
            });
    if (result.wasInterrupted()) {
      return signalPassFailure();
    }

    auto *ctx = &getContext();
    IRRewriter rewriter(ctx);
    for (auto observeOp : observeOps) {
      auto input = observeOp.getInput();
      auto inTy = llvm::cast<MemRefType>(input.getType());
      auto elemTy = inTy.getElementType();

      // Lookup probe function name, and add declaration if needed
      auto probeFuncName = getObserveFuncName(elemTy);
      auto probeFuncTy = getObserveFuncType(ctx, elemTy);
      lookupOrCreateProbeFunc(rewriter, moduleOp, probeFuncName, probeFuncTy);

      rewriter.setInsertionPoint(observeOp);
      auto loc = observeOp.getLoc();
      // Cast to unranked
      auto unrankedInput = memref::CastOp::create(
          rewriter, loc,
          UnrankedMemRefType::get(inTy.getElementType(), kDefaultMemorySpace),
          input);

      // Materialize constants for opID and resultID
      Value opIDVal =
          arith::ConstantOp::create(rewriter, loc, observeOp.getOpIDAttr());
      Value resultIDVal =
          arith::ConstantOp::create(rewriter, loc, observeOp.getResultIDAttr());

      // Insert the call
      rewriter.replaceOpWithNewOp<func::CallOp>(
          observeOp, probeFuncName, TypeRange{},
          ValueRange{unrankedInput, opIDVal, resultIDVal});
    }
  }

  /// Lower probe.report ops in \p funcOp
  void lowerReportOps(ModuleOp moduleOp, func::FuncOp funcOp) {
    std::vector<ReportOp> reportOps;
    funcOp->walk([&](ReportOp op) { reportOps.push_back(op); });

    auto *ctx = &getContext();
    IRRewriter rewriter(ctx);
    auto funcTy = FunctionType::get(ctx, {}, {});
    lookupOrCreateProbeFunc(rewriter, moduleOp, reportFuncName, funcTy);

    for (auto reportOp : reportOps) {
      rewriter.setInsertionPoint(reportOp);
      rewriter.replaceOpWithNewOp<func::CallOp>(reportOp, reportFuncName,
                                                TypeRange{}, ValueRange{});
    }
  }

  /// Lower probe ops in \p funcOp, assuming it is part of \p moduleOp
  void runOnFunction(ModuleOp moduleOp, func::FuncOp funcOp) {
    lowerObserveOps(moduleOp, funcOp);
    lowerReportOps(moduleOp, funcOp);
  }

public:
  using Base::Base;

  void runOnOperation() override {
    auto M = getOperation();
    for (auto F : M.getOps<func::FuncOp>()) {
      runOnFunction(M, F);
    }
  }
};
} // namespace
} // namespace mlir::probe
