#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <random>
#include <vector>

using json = nlohmann::json;

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

cl_context context;
cl_command_queue queue;
cl_program program;

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




    
    return output;

    
}





int main(){
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
    //jetzt einfach einer der test images holen'
    std::ifstream datasetfile ("mnist_handwritten_test.json");
    if(!datasetfile.is_open()){
        std::cout << "fehler beim laden der dataset json" << std::endl;
        return 1;
    }
    //vorherige modelsize wiederherstellen sozusagen
    json datasetJson;
    datasetfile >> datasetJson;
    
    float dataset[modelsize[0]];
    int answer = datasetJson[0]["label"];
    for(int i = 0; i < modelsize[0]; i++)
    {
        dataset[i] = datasetJson[0]["image"][i].get<float>();
    }
    std::cout << answer << std::endl;

    
   
    
    
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
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    //vertig initiallieisert 


    //dort werden immer die zwischen loesungen gespeichert 
    float* uebergangsarray[modellength];
    //test daten input
    for(int i = 0; i < modellength; i++){
        uebergangsarray[i] = new float[modelsize[i]];
    }
    uebergangsarray[0] = dataset;

    
    for(int i = 0; i < modellength - 1; i++){
        uebergangsarray[i + 1] = Forward(biase[i], modelsize[i+ 1], modelsize[i], uebergangsarray[i], weights[i]);
    }
    for(int i = 0; i < modelsize[modellength - 1]; i++){
        std::cout << uebergangsarray[modellength - 1][i] << ' ';
    }
   

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    return 0;
}


