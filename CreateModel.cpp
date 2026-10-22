#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <random>

using json = nlohmann::json;




int modelsize[] = {28*28, 128, 64, 10};
int modellength = sizeof(modelsize) / sizeof(modelsize[0]);


int main(){

    // erstellt ein dreidimensionales array
    float** weights[modellength];
    //und fuer den bias ein zwei dimensionales array
    float* biases[modellength - 1];

    //fuer random number generieren 
    std::random_device rd;
    std::mt19937 gen(rd());
    //bereich festlegen welchen die weights und biases haben sollte es ist eigentlich ziemlich egal sollte halt nur nicht minus sein
    std::uniform_real_distribution<float> dis(0.0f, 0.001f);


    // die oberste array schicht soll so lange sein wie es schichten gibt 
    for(int i = 0; i < modellength - 1; i++){
        // in jeder von dieser oberen shcicht ist eine weitere schicht die so gros ist wie die anzahl an neuronen von der naechsten schicht
        weights[i] = new float*[modelsize[i + 1]];
        
    }
    for(int i = 0; i < modellength - 1; i++){
        for(int j = 0; j < modelsize[i + 1]; j++){
            // inerhalb dieser naechsten mittleren schicht erstllen wir eine letzte chicht die genau so lang ist wie die anzahl der neuronen 
            weights[i][j] = new float[modelsize[i]];
            
            
        }

    }
    // wir haben all das gemacht da sich ein einzigster neuron mit allen anderen neuronen der naechsten schicht verbinded das bedeuted wir haben wenn wir uns auf zwei schichten konzentriert volgende rechnung
    //erste schicht neuronen anzahl * zweite neuronenschicht anzahl 

    for(int i = 0; i < modellength - 1; i++){
        for(int j = 0; j < modelsize[i+1]; j++){
            for(int k = 0; k < modelsize[i]; k++){
                weights[i][j][k] = dis(gen);
            }
        }
    }


    //bei den biases ist es simpler wir brauchen nur die groese der ersten schicht bestimmen was alle hidden layer sind
    for(int i = 0; i < modellength - 1; i++){
        //und die zweite schicht ist einfach so gros wie die anzahl an neuronen
        biases[i] = new float[modelsize[i +1]];
    }
    
    //und jetzt geben wir den biasen noch einen random wert
    for(int i = 0; i <modellength - 1; i++){
        for(int j = 0; j < modelsize[i +1]; j++){
            biases[i][j] = dis(gen);
        }
    }
    
    using namespace nlohmann::literals;


    json modelsizeJson;
    modelsizeJson["modelsize"] = modelsize;
    modelsizeJson["modelsizeLength"] = modellength;
    
    //um es in einer json zu convertieren ist es ein bisschen haesslich und unschoen weil es ein 3d array ist
    json weightsJson = json::array();

    for(int i = 0; i < modellength - 1; i++){
        json jsonLayer = json::array();
        for(int j = 0; j < modelsize[i+1]; j++){
            json jsonRow = json::array();
            for(int k = 0; k < modelsize[i]; k++){
                jsonRow.push_back(weights[i][j][k]);
            }
            jsonLayer.push_back(jsonRow);
        }
        weightsJson.push_back(jsonLayer);
    }

    json biasesJson = json::array();
    for(int i = 0; i <modellength - 1; i++){
        json row = json::array();
        for(int j = 0; j < modelsize[i +1]; j++){
            row.push_back(biases[i][j]);
        }
        biasesJson.push_back(row);
    }

    std::fstream modelsizeFile;
    modelsizeFile.open(R"(modelsize.json)",
        std::ios::out);

    std::fstream weightsFile;
    weightsFile.open(R"(weights.json)",
        std::ios::out);

    std::fstream biaseFile;
    biaseFile.open(R"(biase.json)",
        std::ios::out);

    modelsizeFile << modelsizeJson.dump(4);
    weightsFile << weightsJson.dump(4);
    biaseFile << biasesJson.dump(4);

    

    
    return 0;
    
}