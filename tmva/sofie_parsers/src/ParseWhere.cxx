#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Where.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseSigmoid = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

    for (int i = 0; i < 3; ++i) {
      auto input_name = nodeproto.input(i);
      if (parser.IsRegisteredTensorType(input_name)) {
         // according to ONNX both inputs have same type
         if (i == 1)
            input_type = parser.GetTensorType(input_name);
         else
            if (input_type != parser.GetTensorType(input_name)) {
               throw
                  std::runtime_error("TMVA::SOFIE ONNX parser Binary op has input tensors of different types");
            }
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Binary op has input tensor " + input_name +
                                  " but its type is not yet registered");
      }
   }


  std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Where<float>(nodeproto.input(0), nodeproto.input(1), nodeproto.input(2), output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Where Operator does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA