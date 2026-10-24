#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <random>
#include <vector>
#include <cmath>

using json = nlohmann::json;
//ich habe mir im internet die mnist trainingsdaten als json gedownloaded fuers trainieren gitbs 60 tausend udn fuers testen 10 tausend
const char* kernelSource = R"(
__kernel void array_add(int biassize, int inputsize, __global const float* Bias, __global const float* Input, __global const float* Weights, __global float* Output){

    int i = get_global_id(0);
    if(i < biassize){
        float ubergang = 0.0f;
        for(int j = 0; j < inputsize; j++)
        {
            ubergang += Input[j] * Weights[(i* inputsize) + j];
        }
        Output[i] = 1.0f / (1.0f + exp(-1 * (ubergang + Bias[i])));
    }
}
)";
const char* kernelBackpropagation = R"(
__kernel void array_add(int isend, int previoussize, __global const float* previousDelta,__global const float* PreviousWeights, __global  float* Delta,  int outputsize, __global const float* Output, __global const float* Weights, __global float* BiasGradient, __global float* weightGradient, __global float* inputs,  int  inputsize){
    
    int i = get_global_id(0);
    
    if(i < outputsize){
        float delta = 0.0f;
        if(isend == 1){
        
            
            
            delta += (Output[i] - previousDelta[i]);
            
            delta *= (Output[i] * (1.0f - Output[i]));

            
            BiasGradient[i] = delta;
            
            for (int j = 0; j < inputsize; j++){
                weightGradient[(i * inputsize) + j] = delta * inputs[j];
            }
        }
        else{
            for(int j = 0; j < previoussize; j++ )
            {
                delta += PreviousWeights[(j * outputsize) + i] * previousDelta[j];
            }
            delta *= (Output[i] * (1.0f - Output[i]));

            
            BiasGradient[i] = delta;
            
            for (int j = 0; j < inputsize; j++){
                weightGradient[(i * inputsize) + j] = delta * inputs[j];
            }
        }
        Delta[i] = delta;
        
        
        
        
    }
}
)";

cl_context context;
cl_command_queue queue;
cl_program program;
cl_program program1;

float* Forward(float biase[], int biaseLength,int inputLength, float input[], float** weights){

    


   
    cl_int err;

    

    

    cl_kernel kernel = clCreateKernel(program, "array_add", &err);


    
    size_t biasBytes = sizeof(float) * biaseLength;
    size_t inputBytes = sizeof(float) * inputLength;
    size_t weightsBytes = sizeof(float) * biaseLength * inputLength;
    size_t outputBytes = sizeof(float) * biaseLength;

    float* output = new float[biaseLength];
    //2d weights auf 1d flatten
    float* flatweights = new float[biaseLength * inputLength];
    for(int i = 0; i < biaseLength; i++){
        for(int j = 0; j < inputLength; j++)
        {
            flatweights[(i * inputLength) + j] = weights[i][j];
        }
    }

    //gpu buffers erstellen fuer alle arrays
    cl_mem devbias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, biasBytes, biase, &err);
    cl_mem devInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputBytes, input, &err);
    cl_mem devWeight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightsBytes, flatweights, &err);
    cl_mem devOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY , outputBytes, NULL, &err);
    
    // jetzt tun wir sozusagen die inputs uebergeben also wieder wichtig ist die riehenfolge
    err = clSetKernelArg(kernel, 0, sizeof(int), &biaseLength);
    err = clSetKernelArg(kernel, 1, sizeof(int), &inputLength);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &devbias);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &devInput);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &devWeight);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &devOutput);

    //jetzt fuehren wir das obere program aus
    //sagt basicly wie biele thrats gebraucht werden bei uns ist es 1 pro neuron
    size_t globalWorkSize = biaseLength;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

    clFinish(queue);

    //jetzt das output rauslesen und auf die cpu machen
    err = clEnqueueReadBuffer(queue, devOutput, CL_TRUE, 0, outputBytes, output, 0, NULL, NULL);

    //jetzt noch hinter uns aufrauemen

    clReleaseMemObject(devbias);
    clReleaseMemObject(devInput);
    clReleaseMemObject(devWeight);
    clReleaseMemObject(devOutput);
    clReleaseKernel(kernel);
    delete[] flatweights;
    
    return output;

    
}

float** train(int isend, float previousDelta[],  int biaseLength, float output[], int outputsize,float** weights,  int inputLength, float input[], int previousLength, float** previousWeights, int previousBiasLength, int previousInputLength)
{
    float** returns = new float*[3];
     cl_int err;

    

    

    cl_kernel kernel = clCreateKernel(program1, "array_add", &err);
    float* Delta = new float[outputsize];

    
    size_t previousDeltaBytes = sizeof(float) * previousLength;
    size_t previousweightsBytes = sizeof(float) * previousBiasLength * previousInputLength;
    size_t DeltaByes = sizeof(float) * outputsize;
    size_t inputBytes = sizeof(float) * inputLength;
    size_t outputBytes = sizeof(float) * outputsize;
    size_t weightsbytes = sizeof(float) * inputLength * biaseLength;
    size_t BiasGradiantBytes = sizeof(float) * biaseLength;
    size_t WeightGradiantBytes = sizeof(float) * inputLength * biaseLength;

    float* BiasGradiant = new float[biaseLength];
    float* WeightGradiant = new float[inputLength * biaseLength];
    //2d weights auf 1d flatten
    float* flatweights = new float[biaseLength * inputLength];
    float* flatpreviousWeights = new float[previousBiasLength * previousInputLength];
    for(int i = 0; i < biaseLength; i++){
        for(int j = 0; j < inputLength; j++){
            flatweights[(i * inputLength) + j] = weights[i][j];
        }
    }
    for(int i = 0; i < previousBiasLength; i ++){
        for(int j = 0; j < previousInputLength; j++){
            flatpreviousWeights[(i * previousInputLength) + j] = previousWeights[i][j];
        }
    }

    //gpu buffers erstellen fuer alle arrays
    cl_mem devPreviousDelta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousDeltaBytes, previousDelta, &err);
    cl_mem devpreviousWeight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousweightsBytes, flatpreviousWeights, &err);
    cl_mem devDelta = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DeltaByes, NULL, &err);
    cl_mem devInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputBytes, input, &err);
    cl_mem devWeight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightsbytes, flatweights, &err);
    cl_mem devOutput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, outputBytes, output, &err);
    cl_mem devBiasgradiant = clCreateBuffer(context, CL_MEM_WRITE_ONLY , BiasGradiantBytes, NULL, &err);
    cl_mem devWeightGradiant = clCreateBuffer(context, CL_MEM_WRITE_ONLY , WeightGradiantBytes, NULL, &err);
    
    
    
    // jetzt tun wir sozusagen die inputs uebergeben also wieder wichtig ist die riehenfolge
    err = clSetKernelArg(kernel, 0, sizeof(int), &isend);
    err = clSetKernelArg(kernel, 1, sizeof(int), &previousLength);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &devPreviousDelta);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &devpreviousWeight);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &devDelta);
    err = clSetKernelArg(kernel, 5, sizeof(int), &outputsize);
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &devOutput);
    err = clSetKernelArg(kernel, 7, sizeof(cl_mem), &devWeight);
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &devBiasgradiant);
    err = clSetKernelArg(kernel, 9, sizeof(cl_mem), &devWeightGradiant);
    err = clSetKernelArg(kernel, 10, sizeof(cl_mem), &devInput);
    err = clSetKernelArg(kernel, 11, sizeof(int), &inputLength);

    //jetzt fuehren wir das obere program aus
    //sagt basicly wie biele thrats gebraucht werden bei uns ist es 1 pro neuron
    size_t globalWorkSize = outputsize;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

    clFinish(queue);

    //jetzt das output rauslesen und auf die cpu machen
    err = clEnqueueReadBuffer(queue, devBiasgradiant, CL_TRUE, 0, BiasGradiantBytes, BiasGradiant, 0, NULL, NULL);
    err = clEnqueueReadBuffer(queue, devWeightGradiant, CL_TRUE, 0, WeightGradiantBytes, WeightGradiant, 0, NULL, NULL);
    err = clEnqueueReadBuffer(queue, devDelta, CL_TRUE, 0, DeltaByes, Delta, 0, NULL, NULL);
    //jetzt noch hinter uns aufrauemen

    clReleaseMemObject(devDelta);
    clReleaseMemObject(devPreviousDelta);
    clReleaseMemObject(devpreviousWeight);
    clReleaseMemObject(devInput);
    clReleaseMemObject(devWeight);
    clReleaseMemObject(devOutput);
    clReleaseMemObject(devBiasgradiant);
    clReleaseMemObject(devWeightGradiant);
    clReleaseKernel(kernel);
    delete[] flatweights;
    delete[] flatpreviousWeights;

    returns[0] = BiasGradiant;
    returns[1] = WeightGradiant;
    returns[2] = Delta;
    
    return returns;

}





int main(){
    //einer unserer wichtigsten variablen der loss
    float Loss = 0.8f;
    //hier den lerngradient er dar fnicht zu hoch zu sein aber auch nicht zu fein
    float learngradient = 0.01f;

    
    std::random_device rd;
    std::mt19937 gen(rd());

    
    std::uniform_int_distribution<int> distrib(0, 60000 - 1);


    //zuerst lesen wir die model groese aus
    std::ifstream file ("modelsize.json");
    if(!file.is_open()){
        std::cout << "fehler beim laden der modelsize json" << std::endl;
        return 1;
    }
    //vorherige modelsize wiederherstellen sozusagen
    json modelsizeJson;
    file >> modelsizeJson;
    int modellength = modelsizeJson["modelsizeLength"];
    int modelsize[modellength];
    for(int i = 0; i < modellength; i++){
        modelsize[i] = modelsizeJson["modelsize"][i];
    }
    //jetzt bestimmen wir die weights wieder
    std::ifstream weightsfile ("weights.json");
    if(!weightsfile.is_open()){
        std::cout << "fehler beim laden der weights json" << std::endl;
        return 1;
    }
    json weightsJson;
    weightsfile >> weightsJson;

    float** weights[modellength - 1];
    
    for(int i = 0; i < modellength - 1; i++){
        
        weights[i] = new float*[modelsize[i + 1]];
        
    }
    for(int i = 0; i < modellength - 1; i++){
        for(int j = 0; j < modelsize[i + 1]; j++){
             
            weights[i][j] = new float[modelsize[i]];
            
            
        }

    }
    

    for(int i = 0; i < modellength - 1; i++){
        for(int j = 0; j < modelsize[i+1]; j++){
            for(int k = 0; k < modelsize[i]; k++){
                weights[i][j][k] = weightsJson[i][j][k];
            }
        }
    }

    //und jetzt nur noch die biase 
    std::ifstream biasefile ("biase.json");
    if(!biasefile.is_open()){
        std::cout << "fehler beim laden der biase json" << std::endl;
        return 1;
    }
    json biaseJson;
    biasefile >> biaseJson;
    float* biase[modellength - 1];

    for(int i = 0; i < modellength - 1; i++){
        
        biase[i] = new float[modelsize[i +1]];
    }
    
    
    for(int i = 0; i <modellength - 1; i++){
        for(int j = 0; j < modelsize[i +1]; j++){
            biase[i][j] = biaseJson[i][j];
        }
    }
    //holt die trainings daten
    std::ifstream trainingfile ("mnist_handwritten_train.json");
    if(!trainingfile.is_open()){
        std::cout << "fehler beim laden der train daten json" << std::endl;
        return 1;
    }
    json trainJson;
    trainingfile >> trainJson;
    //wir erstellen einen array den wir immer wieder mit neuen test daten ueberschreiben koenen und einen antwort int
    float dataset[modelsize[0]];
    int answer;



    //open cl initialiersieren bei diesem teil bin ich erlich habe ich ai verwendet wie auch um sachen zu importieren
    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        std::cerr << "Keine OpenCL Plattform gefunden!" << std::endl;
        return 1;
    }
    std::vector<cl::Device> devices;
    if(devices.empty()) {
        for(const auto& availablePlatform : platforms) {
            std::vector<cl::Device> gpuDevices;
            availablePlatform.getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);
            devices.insert(devices.end(), gpuDevices.begin(), gpuDevices.end());
        }
    }
    if(devices.empty()) {
        for(const auto& availablePlatform : platforms) {
            std::vector<cl::Device> platformDevices;
            availablePlatform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
            devices.insert(devices.end(), platformDevices.begin(), platformDevices.end());
        }
    }
    if(devices.empty()) {
        std::cerr << "Kein verwendbares OpenCL-Gerät gefunden!" << std::endl;
        return 1;
    }
    cl::Device device = devices.front();

    cl_device_id raw_device = device.get();
    context = clCreateContext(NULL, 1, &raw_device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device.get(), 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    program1 = clCreateProgramWithSource(context, 1, &kernelBackpropagation, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    clBuildProgram(program1, 0, NULL, NULL, NULL, NULL);
    //vertig initiallieisert 


    //dort werden immer die zwischen loesungen gespeichert 
    float* uebergangsarray[modellength];
    

    for(int u = 0; u < 1; u++){
        float* BiasGradiants[modellength - 1];
        float* weightgradiants[modellength - 1];
        for(int o = 0; o < 60000; o++){
            if(Loss / o >= 0.08f){
                //zuerst bestimmen wir welche trainings date wir holen 
                
                
                for(int i = 0; i < modelsize[0]; i++){
                    dataset[i] = trainJson[o]["image"][i].get<int>() / 255.0f;
                }
                answer = trainJson[o]["label"];
                
                uebergangsarray[0] = dataset;
                

                
                for(int i = 0; i < modellength - 1; i++){
                    uebergangsarray[i + 1] = Forward(biase[i], modelsize[i+ 1], modelsize[i], uebergangsarray[i], weights[i]);
                }
                for(int i = 0; i < modelsize[modellength - 1]; i++){
                    std::cout << uebergangsarray[modellength - 1][i] << ' ';
                }
                
                float modelanswer = uebergangsarray[modellength - 1][answer];
                float UebergangsLoss = 0.0f;
                for (int i = 0; i < modelsize[modellength - 1]; i++) {
                    float target = (i == answer) ? 1.0f : 0.0f;
                    float difference = uebergangsarray[modellength - 1][i] - target;
                    UebergangsLoss += 0.5f * difference * difference;
                }
                Loss += UebergangsLoss;
                std::cout << "modelanswere: " << modelanswer << ", was es haben sollte answer:" << answer << ", loss:" << UebergangsLoss << std::endl;
                //int TrueAnswer, float biase[], int biaseLength, float output[], int outputsize,float** weights,  int inputLength, float input[]
                int isend;
                float** returns;
                
                int outsize = modelsize[modellength - 1];
                float firstDelta[outsize];
                float* currentdelta;
                for(int i = modellength - 1; i > 0; i--){
                    
                    float* inputDelta;
                    
                    if(i == modellength - 1){
                        isend = 1;
                        for(int j = 0; j < modelsize[modellength - 1]; j++){
                            if(j == answer){
                                firstDelta[j] = 1.0f;
                                
                            }
                            else{
                                firstDelta[j] = 0.0f;
                            }
                        }
                        inputDelta = firstDelta;
                        
                        
                    }
                    else{
                        isend = 0;
                        inputDelta = currentdelta;
                        
                        
                    }
                    if(i == modellength - 1){
                        
                        returns = train(isend,inputDelta,modelsize[i], uebergangsarray[i], modelsize[i], weights[i - 1], modelsize[i - 1], uebergangsarray[i - 1], modelsize[i], weights[i - 1], modelsize[i],modelsize[i - 1]);
                    }
                    else{
                        
                        returns = train(isend,inputDelta,modelsize[i], uebergangsarray[i], modelsize[i], weights[i - 1], modelsize[i - 1], uebergangsarray[i - 1], modelsize[i + 1], weights[i ], modelsize[i + 1],modelsize[i]);
                    }

                    if(i != modellength - 1){
                        delete[] currentdelta;
                    }
                    

                    
                    
                    BiasGradiants[i - 1] = returns[0];
                    weightgradiants[i - 1] = returns[1];
                    

                    currentdelta = returns[2];

                    
                    
                }

                //hier applien wir den gradient
                for(int i = modellength - 1; i > 0; i--){
                    for(int j = 0; j < modelsize[i]; j++){
                            
                        biase[i - 1][j] = biase[i- 1][j] - (learngradient * BiasGradiants[i - 1][j]);
                    }
                    for(int j = 0; j < modelsize[i]; j++){
                        for(int k = 0; k < modelsize[i - 1]; k++){
                                
                            weights[i - 1][j][k] = weights[i - 1][j][k] - (learngradient * weightgradiants[i - 1][(j*modelsize[i - 1]) + k]);
                        }
                    }
                }
                delete[] returns[0];
                delete[] returns[1];
                delete[] returns;


                delete[] currentdelta;
                for(int i = 1; i < modellength; i++){
                    delete[] uebergangsarray[i];
                }
            }
            else{
                json weightJson = json::array();
                for(int i = 0; i < modellength - 1; i++){
                    json jsonLayer = json::array();
                    for(int j = 0; j < modelsize[i+1]; j++){
                        json jsonRow = json::array();
                        for(int k = 0; k < modelsize[i]; k++){
                            jsonRow.push_back(weights[i][j][k]);
                        }
                        jsonLayer.push_back(jsonRow);
                    }
                    weightJson.push_back(jsonLayer);
                }

                json biasesJson = json::array();
                for(int i = 0; i <modellength - 1; i++){
                    json row = json::array();
                    for(int j = 0; j < modelsize[i +1]; j++){
                        row.push_back(biase[i][j]);
                    }
                    biasesJson.push_back(row);
                }

                std::fstream weightsFile;
                weightsFile.open(R"(weights.json)",
                    std::ios::out);

                std::fstream biaseFile;
                biaseFile.open(R"(biase.json)",
                    std::ios::out);

                
                weightsFile << weightJson.dump(4);
                biaseFile << biasesJson.dump(4);



                clReleaseCommandQueue(queue);
                clReleaseContext(context);
                clReleaseProgram(program);
                clReleaseProgram(program1);
                return 0;
            }
        }
        Loss = Loss / 60000.0f;
        std::cout << "\n \n \n \n epochen Loss:" << Loss << std::endl;
        Loss = 0.08f;
        
    }   
    
    
    json weightJson = json::array();
    for(int i = 0; i < modellength - 1; i++){
        json jsonLayer = json::array();
        for(int j = 0; j < modelsize[i+1]; j++){
            json jsonRow = json::array();
            for(int k = 0; k < modelsize[i]; k++){
                jsonRow.push_back(weights[i][j][k]);
            }
            jsonLayer.push_back(jsonRow);
        }
        weightJson.push_back(jsonLayer);
    }

    json biasesJson = json::array();
    for(int i = 0; i <modellength - 1; i++){
        json row = json::array();
        for(int j = 0; j < modelsize[i +1]; j++){
            row.push_back(biase[i][j]);
        }
        biasesJson.push_back(row);
    }

    std::fstream weightsFile;
    weightsFile.open(R"(weights.json)",
        std::ios::out);

    std::fstream biaseFile;
    biaseFile.open(R"(biase.json)",
        std::ios::out);

    
    weightsFile << weightJson.dump(4);
    biaseFile << biasesJson.dump(4);



    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseProgram(program1);
    return 0;
}


