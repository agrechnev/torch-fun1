#include <torch/script.h>

#include <iostream>
#include <memory>
#include <vector>

int main(){
    using namespace std;
    
    // Load model
    torch::jit::script::Module module = torch::jit::load("ts_model.pt");
    
    // Inputs
    vector<torch::jit::IValue> inputs{
        torch::ones({1, 1, 28, 28}, at::kCUDA),
    };
    
    // Infer
    at::Tensor outputs = module.forward(inputs).toTensor();
    
    cout << "outputs = " << outputs << endl;
    
    return 0;
}
