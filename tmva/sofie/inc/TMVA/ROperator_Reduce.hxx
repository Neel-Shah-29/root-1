#ifndef TMVA_SOFIE_ROPERATOR_Reduce
#define TMVA_SOFIE_ROPERATOR_Reduce

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum ReduceOpMode { Reduce_mean, Reduce_sumsquare, Reduce_prod };

template <typename T, ReduceOpMode Op1>
struct ReduceOperatorTrait {
   const char *Name() { return ""; }
};
template <typename T>
struct ReduceOperatorTrait <T, Reduce_mean> {
   static const char *Name() { return "Reduce_mean"; }
};

template <typename T>
struct ReduceOperatorTrait <T, Reduce_prod> {
   static const char *Name() { return "Reduce_prod"; }
};

template <typename T>
struct ReduceOperatorTrait <T, Reduce_sumsquare> {
   static const char *Name() { return "Reduce_sumsquare"; }
};

template <typename T, ReduceOpMode Op>
class ROperator_Reduce final : public ROperator
{
private:
    /* Attributes*/
    ReduceOpMode fReduceMode;
    int fkeepdims = 1; //default value
    std::string fNX;
    std::string fNY;
    std::vector<size_t> fShapeX;
    std::vector<size_t> fShapeY;

public:
   // std::string Name() {
   //    if (fReduceMode == Reduce_mean)  return "Reduce_mean";
   //    if (fReduceMode == Reduce_sumsquare)  return "Reduce_sumsquare";
   //    if (fReduceMode == Reduce_prod) return "Reduce_prod"

   //    return "Invalid";
   // }
   ROperator_Reduce(){ }   
   ROperator_Reduce(int keepdims,std::string nameX, std::string nameY):
   fkeepdims(keepdims),fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)) {}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      // assume now inputs have same shape (no broadcasting)
      auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
      return ret;
   }
    void Initialize(RModel& model){

        fUseSession = model.UseSession();

        if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
            throw std::runtime_error("TMVA SOFIE Reduce Op Input Tensor " + fNX + " is not found in model");
        }
        fShapeX = model.GetTensorShape(fNX);
        // find shape of Y and add it in the list of intermediate tensors
        fShapeY = ShapeInference({fShapeX})[0];
        model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
    }

    std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeX.empty() || fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Reduce Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//----  operator " << std::string(ReduceOperatorTrait<T,Op>::Name()) << "  " << OpName << "\n";
      out << "{\n"; // create a new scope to avoid name clash
      size_t length = ConvertShapeToLength(fShapeX);

      out << SP << "int " << OpName << "_keepdims = " <<  fkeepdims << ";\n";

      if(fkeepdims == 1){
         if(fReduceMode == Reduce_mean){
         out << SP << "double sum = 0.0;\n";
         out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
         out << SP << SP << "sum += tensor_" << fNX << "[id];\n";
         out << SP << "}\n";
         out << SP << "tensor_" << fNY << "= sum/" << length << ";\n";
         }
         else if(fReduceMode == Reduce_sumsquare){
            out << SP << "double sum = 0.0;\n";
            out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
            out << SP << SP << "sum += tensor_" << fNX << "[id]* tensor_" << fNX << "[id] ;\n";
            out << SP << "}\n";
            out << SP << "tensor_" << fNY << "= sum/" << length << ";\n";
         }
         else if(fReduceMode == Reduce_prod){
            out << SP << "double sum = 0.0;\n";
            out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
            out << SP << SP << "sum *= tensor_" << fNX << "[id];\n";
            out << SP << "}\n";
            out << SP << "tensor_" << fNY << "= sum/" << length << ";\n";
         }
      }
      // end scope
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Reduce
