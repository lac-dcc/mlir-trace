#ifndef PROBE_INCLUDE_DIALECT_PROBE_IR_PROBEOPS_TD
#define PROBE_INCLUDE_DIALECT_PROBE_IR_PROBEOPS_TD

include "ProbeBase.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Probe_Op<string mnemonic, list<Trait> traits = []>
    : Op<Probe_Dialect, mnemonic, traits>;

def Probe_ObserveOp: Probe_Op<"observe", [
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "getAliasingValues",
      "bufferize"
    ]>]> {
  let summary = "Observe runtime values for an ND tensor";
  let description = [{
    Operation that observes runtime values of ND tensors at runtime. This
    operation is lowered to a function call to a runtime library function,
    which can apply any kind of side-effect-free probing.
  }];
  let hasVerifier = 1;

  let arguments = (ins
      AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$input,
      I32Attr:$opID,
      I32Attr:$resultID
  );
  let results = (outs);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict
  }];

  // Explicitly implement bufferization
  let extraClassDefinition = [{
    bool ObserveOp::bufferizesToMemoryRead(
        ::mlir::OpOperand &operand,
        const ::mlir::bufferization::AnalysisState &state) {
      return true;
    }

    bool ObserveOp::bufferizesToMemoryWrite(
        ::mlir::OpOperand &operand,
        const ::mlir::bufferization::AnalysisState &state) {
      return false;
    }

    ::mlir::bufferization::AliasingValueList
    ObserveOp::getAliasingValues(
        ::mlir::OpOperand &opOperand,
        const ::mlir::bufferization::AnalysisState &state) {
      return {};
    }

    llvm::LogicalResult
    ObserveOp::bufferize(
        ::mlir::RewriterBase &rewriter,
        const ::mlir::bufferization::BufferizationOptions &options,
        ::mlir::bufferization::BufferizationState &state) {
      auto inputBufferOrFailure =
          ::mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
      if (failed(inputBufferOrFailure))
        return llvm::failure();

      auto tensorOp = getOperation();
      auto memrefOp = ObserveOp::create(rewriter, getLoc(),
          *inputBufferOrFailure, getOpIDAttr(), getResultIDAttr());
      memrefOp->setAttrs(tensorOp->getAttrs());
      rewriter.replaceOp(tensorOp, memrefOp);
      return llvm::success();
    }
  }];
}

def Probe_ReportOp: Probe_Op<"report"> {
  let summary = "Report observations that have been made so far";
  let description = [{
    Operation that reports information that has been observed up until the
    point it is invoked. The format which is used to report information can be
    customized by the implementer.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = [{
    `(` `)` attr-dict
  }];
}

#endif // PROBE_INCLUDE_DIALECT_PROBE_IR_PROBEOPS_TD
