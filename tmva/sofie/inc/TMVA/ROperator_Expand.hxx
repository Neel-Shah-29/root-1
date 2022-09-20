#ifndef TMVA_SOFIE_ROPERATOR_Expand
#define TMVA_SOFIE_ROPERATOR_Expand

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"
#include <string>
#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Expand final : public ROperator
{

private:

   std::string fNX;
   std::vector<size_t> fNew_Shape;
   std::string fNY;
   std::vector<size_t> fShapeX;

public:
   ROperator_Expand(){}
   ROperator_Expand(std::string nameX, std::vector<size_t> new_Shape, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNew_Shape(new_Shape), fNY(UTILITY::Clean_name(nameY)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      std::vector<std::vector<size_t>> ret;
      for(auto it:fNew_Shape)
        ret[0].push_back(it);
      return ret;
   }

   void Initialize(RModel& model){
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("TMVA SOFIE Expand Op Input Tensor is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fNew_Shape);

   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA SOFIE Expand called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShapeX);
      out << "\n//------ EXPAND\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      //Copy all elements of fNX to fNY
      out << SP << SP << "tensor_" << fNY << "[id] = tensor_" << fNX << "[id];\n";
      out << SP << "}\n";
      if(fShapeX.size() == fNew_Shape.size() == 2){
         for(int i = 0; i < fShapeX.size();i++){
            if(fShapeX[i] < fNew_Shape[i]){
               //Condition to copy the rows or columns as required
               
               if(fShapeX[0]<fNew_Shape[0]){
                  
                  //Expand the rows
                  int diff = fNew_Shape[0] - fShapeX[0];
                  for(int j = 0; j < diff; j++){
                     for(int k = length; k < length + fShapeX[1]; k++)
                        out << SP << SP << "tensor_" << fNY << ".push_back(tensor_" << fNX << "[" << k - fShapeX[1] - 1 <<"]);\n";
                  }
               }
               else if(fShapeX[1]<fNew_Shape[1]){
                  //Copy the elements rowwise and insert elements required at the  correct position in between.
                  int diff = fNew_Shape[1] - fShapeX[1];
                  for(int m = 0; m < fShapeX[0]; m++){
                     for(int n = 0; n < diff; n++){
                        out << SP << SP << "tensor_" << fNY << ".insert(" << fShapeX[1] * m << ",tensor_" << fNX[m*fShapeX[1]] << ");\n";
                     }
                  }
               }
            }

         }

      }
      else if (fShapeX.size() == fNew_Shape.size()){
            throw std::runtime_error("TMVA SOFIE Expand support only for 2 dimensional tensors");   
         }
      else if(fShapeX.size()!=fNew_Shape.size()){
         //Dimensions of output tensor is different then the input tensor
         throw std::runtime_error("TMVA SOFIE Expand support only tensors with same input and output dimensions");  
      }
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Expand