//XOR Problem in C
//-------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h> 

//Constant Declaration
#define learning_rate 0.1

#define n_inputs 2                                      //# of Input Nodes
#define n_hidden_1 4                                    //# of Hidden Nodes 
#define n_hidden_2 4                                    //...
#define n_outputs 2                                     //# of Output Nodes

//(2 Hidden Layer) Neural Network Structure
typedef struct {
    double input_layer[n_inputs];                            //Activations
    double hidden_layer_1[n_hidden_1];                       //,,,
    double hidden_layer_2[n_hidden_2];                       //...
    double output_layer[n_outputs];                          //...
    
    double hidden_layer_1_bias[n_hidden_1];                  //Bias
    double hidden_layer_2_bias[n_hidden_2];                  //...
    double output_layer_bias[n_outputs];                     //...
    
    double hidden_weights_1[n_inputs][n_hidden_1];           //Weights
    double hidden_weights_2[n_hidden_1][n_hidden_2];         //...
    double output_weights[n_hidden_2][n_outputs];            //...
} Neural_Network;

typedef struct {
    double hidden_weights_1_grad[n_inputs][n_hidden_1];      //Weights Gradients
    double hidden_weights_2_grad[n_hidden_1][n_hidden_2];    //...
    double output_weights_grad[n_hidden_2][n_outputs];       //...

    double hidden_layer_1_bias_grad[n_hidden_1];             //Bias Gradients
    double hidden_layer_2_bias_grad[n_hidden_2];             //...
    double output_layer_bias_grad[n_outputs];                //...

    double softmax_Outputs[n_outputs];                       //Stored Softmax Outputs
} label_Grad;

//Global Initalization
Neural_Network N_Network;

//Settings
char activation_Type = 'S';                              //Activation Type: S for Sigmoid, R for ReLu, L for Leaky ReLu
char speculative_Backpropagation = 'N';                  //Speculative Backpropagation? Y for Yes, N for No
float threshold = 0.25;

//-------------------------------------------------------
//Activation Functions
double sigmoid(double x){
  return 1/(1+exp(-x));
}

double sigmoid_D(double x){
  return sigmoid(x)*(1-sigmoid(x));
}

double re_lu(double x){
  if(x > 0){
    return x;
  }
  else{
    return 0;
  }
}

double re_lu_D(double x){
  if(x <= 0){
    return 0;
  }
  else{
    return 1;
  }
}

void compute_Softmax_Output(){ 
  double sum = 0.0;
  for(int i = 0; i < n_outputs; i++){
    sum += exp(N_Network.output_layer[i]);
  }

  for(int j = 0; j < n_outputs; j++){
    N_Network.output_layer[j] = exp(N_Network.output_layer[j])/sum;
  }
}

//-------------------------------------------------------
//Neural Network Functions
void init_neural_network(){
  //Randomize Weights ---
  //Randomizes HD1 Weights
  for(int i = 0; i<n_inputs;i++){
    for(int j = 0; j<n_hidden_1; j++){
      N_Network.hidden_weights_1[i][j] = ((double)rand())/((double)RAND_MAX);         //Random Number/(Max of Random(32767)) (0-1)
    }
  }

  //Randomizes HD2 Weights
  for(int i = 0; i<n_hidden_1;i++){
    for(int j = 0; j<n_hidden_2; j++){
      N_Network.hidden_weights_2[i][j] = ((double)rand())/((double)RAND_MAX);
    }
  }

  //Randomizes Output Weights
  for(int i = 0; i<n_hidden_2;i++){
    for(int j = 0; j<n_outputs; j++){
      N_Network.output_weights[i][j] = ((double)rand())/((double)RAND_MAX);
    }
  }

  //Randomize Bias ---
  //Randomizes HD1 Bias
  for(int i = 0; i<n_hidden_1;i++){
      N_Network.hidden_layer_1_bias[i] = ((double)rand())/((double)RAND_MAX);
  }

    //Randomizes HD2 Bias
  for(int i = 0; i<n_hidden_2;i++){
      N_Network.hidden_layer_2_bias[i] = ((double)rand())/((double)RAND_MAX);
  }

    //Randomizes HD1 Bias
  for(int i = 0; i<n_outputs;i++){
      N_Network.output_layer_bias[i] = ((double)rand())/((double)RAND_MAX);
  }
}

void init_neural_network_Xavier(){
  double max = sqrt( (6.0 / (n_inputs + n_hidden_1)) );
  for (int i = 0; i < n_inputs;i++){
    for(int j = 0; j < n_hidden_1; j++){
      N_Network.hidden_weights_1[i][j] = -max + ((float)rand() / RAND_MAX) * (max + max);
    }
  }

  max = sqrt( (6.0 / (n_hidden_1 + n_hidden_2)) );
  for (int i = 0; i < n_hidden_1;i++){
    for(int j = 0; j < n_hidden_2; j++){
      N_Network.hidden_weights_2[i][j] = -max + ((float)rand() / RAND_MAX) * (max + max);
    }
  }

  max = sqrt( (6.0 / (n_hidden_2 + n_outputs)) );
  for (int i = 0; i < n_hidden_2;i++){
    for(int j = 0; j < n_outputs; j++){
      N_Network.output_weights[i][j] = -max + ((float)rand() / RAND_MAX) * (max + max);
    }
  }

  for(int i = 0; i < n_hidden_1; i++){
    N_Network.hidden_layer_1_bias[i] = 0;
  }

  for(int i = 0; i < n_hidden_2; i++){
    N_Network.hidden_layer_2_bias[i] = 0;
  }

  for(int i = 0; i < n_outputs; i++){
    N_Network.output_layer_bias[i] = 0;
  }

}

void feed_Forward(){
  //Get HDL1 Activation
  for(int i = 0; i<n_hidden_1;i++){
    double sum = 0;
    for(int j = 0;j<n_inputs;j++){
      sum+=N_Network.input_layer[j]*N_Network.hidden_weights_1[j][i];
    }
    if(activation_Type == 'R'){
      N_Network.hidden_layer_1[i] = re_lu(N_Network.hidden_layer_1_bias[i] + sum);
    }
    else if(activation_Type == 'S'){
      N_Network.hidden_layer_1[i] = sigmoid(N_Network.hidden_layer_1_bias[i] + sum);
    }
  }

  //Get HDL2 Activation
  for(int i = 0; i<n_hidden_2;i++){
    double sum = 0;
    for(int j = 0;j<n_hidden_1;j++){
      sum+=N_Network.hidden_layer_1[j]*N_Network.hidden_weights_2[j][i];
    }
    if(activation_Type == 'R'){
      N_Network.hidden_layer_2[i] = re_lu(N_Network.hidden_layer_2_bias[i] + sum);
    }
    else if(activation_Type == 'S'){
      N_Network.hidden_layer_2[i] = sigmoid(N_Network.hidden_layer_2_bias[i] + sum);
    }
  }

  //Get Output Layer Activation
  for(int i = 0; i<n_outputs;i++){
    double sum = 0;
    for(int j = 0;j<n_hidden_2;j++){
      sum+=N_Network.hidden_layer_2[j]*N_Network.output_weights[j][i];
    }
    if(activation_Type == 'R'){
      N_Network.output_layer[i] = re_lu(N_Network.output_layer_bias[i] + sum);
    }
    else if(activation_Type == 'S'){
      N_Network.output_layer[i] = sigmoid(N_Network.output_layer_bias[i] + sum);
    }
  }
}

void backpropagation_xor(double desired[])
{
  //Calculating Error Deltas
  //Gives you the Output Error Delta
  double o_error[n_outputs];
  for(int i = 0;i<n_outputs;i++){
    if(activation_Type == 'R'){
      o_error[i]=(desired[i]-N_Network.output_layer[i])*re_lu_D(N_Network.output_layer[i]);
    }
    else if(activation_Type == 'S'){
      o_error[i]=(desired[i]-N_Network.output_layer[i])*sigmoid_D(N_Network.output_layer[i]);
    }
  } //Gives you the error delta of the output layer
  //Gives you the Hidden Layer 2 Error Delta
  double H2_error[n_hidden_2];
  for(int i = 0;i<n_hidden_2;i++){
    double sum = 0;
    for (int j = 0;j<n_outputs;j++){
      sum += o_error[j] * N_Network.output_weights[i][j]; //Sums the Effect of Each output node on the current node
    }
    if(activation_Type == 'R'){
      H2_error[i] = sum*re_lu_D(N_Network.hidden_layer_2[i]); //Finds the error delta after the sum
    }
    else if(activation_Type == 'S'){
      H2_error[i] = sum*sigmoid_D(N_Network.hidden_layer_2[i]); //Finds the error delta after the sum
    }

  }
  //Gives you the Hidden Layer 1 Error Delta
  double H1_error[n_hidden_1];
  for(int i = 0;i<n_hidden_1;i++){
    double sum = 0;
    for (int j = 0; j<n_hidden_2;j++){
      sum += H2_error[j] * N_Network.hidden_weights_2[i][j]; 
    }
    if(activation_Type == 'R'){
      H1_error[i] = sum*re_lu_D(N_Network.hidden_layer_1[i]); 
    }
    else if(activation_Type == 'S'){
      H1_error[i] = sum*sigmoid_D(N_Network.hidden_layer_1[i]); 
    }
  }
  //Currently Stuck Here <--------------
  //Calculating the Adjusted Weights and Bias
  //Adjusts the Hidden_2-Output Weights and Bias
  for (int i = 0; i<n_outputs;i++){
    N_Network.output_layer_bias[i] += o_error[i]*learning_rate;
    for (int j = 0; j<n_hidden_2;j++){
      N_Network.output_weights[j][i] += learning_rate*o_error[i]*N_Network.hidden_layer_2[j];
    }
  }
  //Adjusts the Hidden_1-Hidden_2 Weights and Bias
  for (int i = 0; i<n_hidden_2;i++){
    N_Network.hidden_layer_2_bias[i] += H2_error[i]*learning_rate;
    for (int j = 0; j<n_hidden_1;j++){
      N_Network.hidden_weights_2[j][i] += learning_rate*H2_error[i]*N_Network.hidden_layer_1[j];
    }
  }
  //Adjusts the Input-Hidden_1 Weights and Bias
  for (int i = 0; i<n_hidden_1;i++){
    N_Network.hidden_layer_1_bias[i] += H1_error[i]*learning_rate;
    for (int j = 0; j<n_inputs;j++){
      N_Network.hidden_weights_1[j][i] += learning_rate*H1_error[i]*N_Network.input_layer[j];
    }
  }

}

int get_Max(double arr[]){
  double max = arr[0];
  int enume = 0;
  for(int i = 1; i < n_outputs; i++){                                             //Loops through code looking for the highest number, saves the index
    if(max < arr[i]){
      max = arr[i];
      enume = i;
    }
  }
  return enume;
}

void train_xor(int data[],int correct[],int data_num,int epochs){
  for(int f = 0; f<epochs; f++){
    printf("\nEpoch: %d\n",f);
    double desired[n_outputs];

    for(int i = 0; i < data_num; i++){

      for(int j = 0; j < 2;j++){
        N_Network.input_layer[j] = data[j+i*n_inputs];
      }
      
      if(correct[i] == 1){
        desired[0] = 0;
        desired[1] = 1;
      }
      else{
        desired[0] = 1;
        desired[1] = 0;
      }

      feed_Forward();
      backpropagation_xor(desired);
    }
  }
}

void test_xor(int images[], int labels[]){
  int max_Pos = 0;
  double count = 0;
  for(int k = 0;k<4;k++){
    for(int j = 0; j < n_inputs;j++){
        N_Network.input_layer[j] = images[j+k*n_inputs];
      }
    feed_Forward();
    max_Pos = get_Max(N_Network.output_layer);
    if(max_Pos == labels[k]){
      count++; 
    }
    printf("\n%d %d -> %d",images[k*n_inputs],images[k*n_inputs+1],max_Pos);
  }
  printf("\nCorrect Rate: %f",count/4.0);
  printf("\n");
  for(int i = 0;i<n_outputs;i++){
    printf("%f ",N_Network.output_layer[i]);
  }
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//MAIN

int main()
{
    //Data
    int input_data[4*2] = {
      0,0,
      1,1,
      1,0,
      0,1
    };

    int input_correct[4] = {
      0,0,1,1
    };

    //Random Initalization
    srand((unsigned int)time(NULL));                                        //Generate Random Seed for network initialization

    init_neural_network();                                                 //Initializes Neural Network with Random Values

    int epochs = 100000;
    time_t start, end;
    start = clock();
    if(speculative_Backpropagation == 'N'){
      train_xor(input_data,input_correct,4,epochs);
    }
    end = clock();
    test_xor(input_data,input_correct);
    printf("\nEpochs: %d\n",epochs);
    printf("\nElapsed Time: %f seconds\n",(double)(end-start)/CLOCKS_PER_SEC);

    return 0;
}

