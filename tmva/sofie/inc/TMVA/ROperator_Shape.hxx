#ifndef TMVA_SOFIE_ROPERATOR_Shape
#define TMVA_SOFIE_ROPERATOR_Shape

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <iostream>
#include<sstream>
#include<vector>
#include <iterator>
#include<string>
namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Shape final : public ROperator
{

private:

   /* Attributes*/
   // int start = 0; //default value
   // int end ;
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::string fType;

public:
   ROperator_Shape(){}
   ROperator_Shape(std::string nameX, std::string nameY):
   fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Shape Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Shape op called to Generate without being initialized first");
      }
      std::stringstream out;
      
      // size_t length = ConvertShapeToLength(fShape);
      std::stringstream result;
      std::copy(fShape.begin(), fShape.end(), std::ostream_iterator<int>(result, ""));

      // out << SP << "int " << OpName << "_start = "  << fstart << ";\n";
      // out << SP << "int " << OpName << "_end = "  << fend << ";\n";
   
      out << "\n//------ Shape\n";
      out << SP << "for (int id = 0; id < " << fShape.size() << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = " << result.str() << "[id] ;\n";
      out << SP << "}\n";
      return out.str();
   }
   
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Shape
