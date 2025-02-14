//Character Recognition in C
//-------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>

//Constant Declaration
#define learning_rate 0.01

#define n_inputs 784                                    //# of Input Nodes
#define n_hidden_1 35                                   //# of Hidden Nodes 
#define n_hidden_2 35                                   //...
#define n_outputs 10                                    //# of Output Nodes

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

typedef struct {
    double input_layer_ex[n_inputs];                         //Activations
    double hidden_layer_1_ex[n_hidden_1];                    //,,,
    double hidden_layer_2_ex[n_hidden_2];                    //...
    double output_layer_ex[n_outputs];                       //...
} extra_Act;

//Global Initalization
Neural_Network N_Network;
extra_Act stored_Act;
label_Grad speculative_Label_Array[n_outputs];
int set_Grad[n_outputs] = {0};
int below_Thres_Flag = 0;

//Settings
char activation_Type = 'R';                              //Activation Type: S for Sigmoid, R for ReLu, L for Leaky ReLu
char speculative_Backpropagation = 'Y';                  //Speculative Backpropagation? Y for Yes, N for No
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

double cross_entropy_loss(double soft_prev[],double soft_cur[]){
  double sum = 0;
  for(int i = 0;i<n_outputs;i++){
    sum += -soft_cur[i]*log(soft_prev[i] + 1e-20);
  }
  return sum;
}

//-------------------------------------------------------
//MNIST Data Manipulation
void load_MNIST_Images(char *address, double **images){
  FILE *fp = fopen(address, "rb");                                                //Creates a File Pointer to the Image File (read byte mode)
  if(fp == NULL){                                                                 //Checking to make sure File was read in Correctly
    printf("Error Opening File");
    fclose(fp);
    return;
  }

  int magic_Number = 0;                                                           //Taking the Magic Number out of the File Stream
  fread(&magic_Number, sizeof(int), 1, fp);

  int image_Num = 0;                                      
  fread(&image_Num,sizeof(int),1,fp);                                             //Extracts the number of labels
  image_Num = __builtin_bswap32(image_Num);                                       //Flips the byte order since it comes in reverse
  printf("Address Location: %s has %d Labels.\n", address,image_Num);             //Prints the number of labels in the File

  int rows = 0;
  fread(&rows,sizeof(int),1,fp);                                                  //Extracts the number of rows
  rows = __builtin_bswap32(rows);  

  int columns = 0;
  fread(&columns,sizeof(int),1,fp);                                               //Extracts the number of columns
  columns = __builtin_bswap32(columns);  
  
  int image_Size = image_Num*rows*columns;

  unsigned char *temp = (unsigned char*)malloc(image_Size*sizeof(unsigned char)); //Allocates space for a temporary array that take in the bytes
  fread(temp,sizeof(unsigned char),image_Size,fp);                                //Reads the images into the array
  fclose(fp);

  *images = (double*)malloc(image_Size*sizeof(double));                           //Allocate the space for the Image Array in double form (0-1)

  for(int i = 0; i < image_Size; i++){                                             
    *(*images + i) = temp[i]/ 255.0;                                              //Fits the data between 0-1 and puts it into Image (Has to be decimal 255.0)
  }

  free(temp);                                                                     //Free the allocated space for the array

}

void load_MNIST_Labels(char *address,unsigned char **labels){    
  FILE *fp = fopen(address,"rb");                                                 //Creates a File Pointer to the Label File (read byte mode)
  if(fp == NULL){                                                                 //Checking to make sure File was read in correctly
    printf("Error Opening File");
    fclose(fp);
    return;
  }

  int magic_Number = 0;
  fread(&magic_Number, sizeof(int),1,fp);                                         //Taking the Magic Number out of the File Stream
  
  int label_num = 0;
  fread(&label_num,sizeof(int),1,fp);                                             //Extracts the number of labels
  label_num = __builtin_bswap32(label_num);                                       //Flips the byte order since it comes in reverse
  printf("Address Location: %s has %d Labels.\n",address,label_num);              //Prints the Number of Labels in the File

  *labels = (unsigned char*)malloc(label_num*sizeof(unsigned char));              //Allocates the training labels array to have enough space for the labels
  fread(*labels,sizeof(unsigned char),label_num,fp);                              //Reads the labels into the training labels array

  fclose(fp);
}

void print_MNIST_Images(double image[],int img_Num){
 printf("\n---\n");
  for(int i = 0; i<28;i++){                                                       //Will go row by row and put a value depending on the input number
    printf("\n");
    for(int j = 0; j<28; j++){
      double cur = image[j + i*28+img_Num*784];                           
      if(cur == 0){
          printf("  ");
      }
      else if(cur >= 0.25 && cur < 0.5){
          printf("- ");
      }
      else if(cur >= 0.5 && cur < 0.75){
          printf("+ ");
      }
      else{
          printf("O ");
      }
    }
  }
  printf("\n---\n");
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

  compute_Softmax_Output();
}

void feed_Forward_spec(){
  //Get HDL1 Activation
  for(int i = 0; i<n_hidden_1;i++){
    double sum = 0;
    for(int j = 0;j<n_inputs;j++){
      sum+=N_Network.input_layer[j]*N_Network.hidden_weights_1[j][i];
    }
    N_Network.hidden_layer_1[i] = re_lu(N_Network.hidden_layer_1_bias[i] + sum);
  }

  //Get HDL2 Activation
  for(int i = 0; i<n_hidden_2;i++){
    double sum = 0;
    for(int j = 0;j<n_hidden_1;j++){
      sum+=N_Network.hidden_layer_1[j]*N_Network.hidden_weights_2[j][i];
    }
    N_Network.hidden_layer_2[i] = re_lu(N_Network.hidden_layer_2_bias[i] + sum);
  }

  //Get Output Layer Activation
  for(int i = 0; i<n_outputs;i++){
    double sum = 0;
    for(int j = 0;j<n_hidden_2;j++){
      sum+=N_Network.hidden_layer_2[j]*N_Network.output_weights[j][i];
    }
    N_Network.output_layer[i] = re_lu(N_Network.output_layer_bias[i] + sum);
  }
  compute_Softmax_Output();
}

void backpropagation(double desired[],int label)
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
    if(speculative_Backpropagation == 'Y'){
      speculative_Label_Array[label].output_layer_bias_grad[i] = o_error[i]*learning_rate;//Saves Out Bias for Speculation
    }
    for (int j = 0; j<n_hidden_2;j++){
      N_Network.output_weights[j][i] += learning_rate*o_error[i]*N_Network.hidden_layer_2[j];
      if(speculative_Backpropagation == 'Y'){
        speculative_Label_Array[label].output_weights_grad[j][i] = learning_rate*o_error[i]*N_Network.hidden_layer_2[j]; //Saves Out Weights for Speculation
      }
    }
  }
  //Adjusts the Hidden_1-Hidden_2 Weights and Bias
  for (int i = 0; i<n_hidden_2;i++){
    N_Network.hidden_layer_2_bias[i] += H2_error[i]*learning_rate;
    if(speculative_Backpropagation == 'Y'){
      speculative_Label_Array[label].hidden_layer_2_bias_grad[i] = H2_error[i]*learning_rate; //Saves H2 Bias for Speculation
    }
    for (int j = 0; j<n_hidden_1;j++){
      N_Network.hidden_weights_2[j][i] += learning_rate*H2_error[i]*N_Network.hidden_layer_1[j];
      if(speculative_Backpropagation == 'Y'){
        speculative_Label_Array[label].hidden_weights_2_grad[j][i] = learning_rate*H2_error[i]*N_Network.hidden_layer_1[j]; //Saves H2 Weights for Speculation
      }
    }
  }
  //Adjusts the Input-Hidden_1 Weights and Bias
  for (int i = 0; i<n_hidden_1;i++){
    N_Network.hidden_layer_1_bias[i] += H1_error[i]*learning_rate;
    if(speculative_Backpropagation == 'Y'){
      speculative_Label_Array[label].hidden_layer_1_bias_grad[i] = H1_error[i]*learning_rate; //Saves H2 Bias for Speculation
    }
    for (int j = 0; j<n_inputs;j++){
      N_Network.hidden_weights_1[j][i] += learning_rate*H1_error[i]*N_Network.input_layer[j];
      if(speculative_Backpropagation == 'Y'){
        speculative_Label_Array[label].hidden_weights_1_grad[j][i] = learning_rate*H1_error[i]*N_Network.input_layer[j]; //Saves H2 Weights for Speculation
        set_Grad[label] = 1; //A little redudant, will fix later
      }
    }
  }

}

void backpropagation2(double desired[],int label)
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
    speculative_Label_Array[label].output_layer_bias_grad[i] = o_error[i]*learning_rate;//Saves Out Bias for Speculation
    for (int j = 0; j<n_hidden_2;j++){
      N_Network.output_weights[j][i] += learning_rate*o_error[i]*N_Network.hidden_layer_2[j];   
      speculative_Label_Array[label].output_weights_grad[j][i] = learning_rate*o_error[i]*N_Network.hidden_layer_2[j]; //Saves Out Weights for Speculation
    }
  }
  //Adjusts the Hidden_1-Hidden_2 Weights and Bias
  for (int i = 0; i<n_hidden_2;i++){
    N_Network.hidden_layer_2_bias[i] += H2_error[i]*learning_rate;
    speculative_Label_Array[label].hidden_layer_2_bias_grad[i] = H2_error[i]*learning_rate; //Saves H2 Bias for Speculation
    for (int j = 0; j<n_hidden_1;j++){
      N_Network.hidden_weights_2[j][i] += learning_rate*H2_error[i]*N_Network.hidden_layer_1[j];
      speculative_Label_Array[label].hidden_weights_2_grad[j][i] = learning_rate*H2_error[i]*N_Network.hidden_layer_1[j]; //Saves H2 Weights for Speculation
    }
  }
  //Adjusts the Input-Hidden_1 Weights and Bias
  for (int i = 0; i<n_hidden_1;i++){
    N_Network.hidden_layer_1_bias[i] += H1_error[i]*learning_rate;
    speculative_Label_Array[label].hidden_layer_1_bias_grad[i] = H1_error[i]*learning_rate; //Saves H2 Bias for Speculation
    for (int j = 0; j<n_inputs;j++){
      N_Network.hidden_weights_1[j][i] += learning_rate*H1_error[i]*N_Network.input_layer[j];
        speculative_Label_Array[label].hidden_weights_1_grad[j][i] = learning_rate*H1_error[i]*N_Network.input_layer[j]; //Saves H2 Weights for Speculation
    }
  }

}

void backpropagation_spec(double desired[],int label)
{
  //Output Layer Error
  double o_error[n_outputs];
  for(int i = 0;i<n_outputs;i++){
    o_error[i]=(desired[i]-stored_Act.output_layer_ex[i])*re_lu_D(stored_Act.output_layer_ex[i]);
  } 

  //Hidden Layer 2 Error
  double H2_error[n_hidden_2];
  for(int i = 0;i<n_hidden_2;i++){
    double sum = 0;
    for (int j = 0;j<n_outputs;j++){
      sum += o_error[j] * N_Network.output_weights[i][j]; //Sums the Effect of Each output node on the current node
    } 
    H2_error[i] = sum*re_lu_D(stored_Act.hidden_layer_2_ex[i]); //Finds the error delta after the sum
  }

  //Hidden Layer 1 Error
  double H1_error[n_hidden_1];
  for(int i = 0;i<n_hidden_1;i++){
    double sum = 0;
    for (int j = 0; j<n_hidden_2;j++){
      sum += H2_error[j] * N_Network.hidden_weights_2[i][j]; 
    }
    H1_error[i] = sum*re_lu_D(stored_Act.hidden_layer_1_ex[i]); 
  }


  //Gradients for Output Weights/Bias
  for (int i = 0; i<n_outputs;i++){
    speculative_Label_Array[label].output_layer_bias_grad[i] = o_error[i]*learning_rate;//Saves Out Bias for Speculation
    for (int j = 0; j<n_hidden_2;j++){
      speculative_Label_Array[label].output_weights_grad[j][i] = learning_rate*o_error[i]*stored_Act.hidden_layer_2_ex[j]; //Saves Out Weights for Speculation
    }
  }

  //Gradients for Hidden Layer 2 Weights/Bias
  for (int i = 0; i<n_hidden_2;i++){
    speculative_Label_Array[label].hidden_layer_2_bias_grad[i] = H2_error[i]*learning_rate; //Saves H2 Bias for Speculation
    for (int j = 0; j<n_hidden_1;j++){
      speculative_Label_Array[label].hidden_weights_2_grad[j][i] = learning_rate*H2_error[i]*stored_Act.hidden_layer_1_ex[j]; //Saves H2 Weights for Speculation
    }
  }

  //Gradients for Hidden Layer 1 Weights/Bias
  for (int i = 0; i<n_hidden_1;i++){
    speculative_Label_Array[label].hidden_layer_1_bias_grad[i] = H1_error[i]*learning_rate; //Saves H2 Bias for Speculation
    for (int j = 0; j<n_inputs;j++){
      speculative_Label_Array[label].hidden_weights_1_grad[j][i] = learning_rate*H1_error[i]*stored_Act.input_layer_ex[j]; //Saves H2 Weights for Speculation
    }
  }
}

void s_backpropagation(int label){
  //Adjust Output Weights and Bias
  for (int i = 0; i<n_outputs;i++){
    N_Network.output_layer_bias[i] += speculative_Label_Array[label].output_layer_bias_grad[i];
    for (int j = 0; j<n_hidden_2;j++){
      N_Network.output_weights[j][i] += speculative_Label_Array[label].output_weights_grad[j][i];
    }
  }

  //Adjusts the Hidden_1-Hidden_2 Weights and Bias
  for (int i = 0; i<n_hidden_2;i++){
    N_Network.hidden_layer_2_bias[i] += speculative_Label_Array[label].hidden_layer_2_bias_grad[i];
    for (int j = 0; j<n_hidden_1;j++){
      N_Network.hidden_weights_2[j][i] += speculative_Label_Array[label].hidden_weights_2_grad[j][i];
    }
  }

  //Adjusts the Input-Hidden_1 Weights and Bias
  for (int i = 0; i<n_hidden_1;i++){
    N_Network.hidden_layer_1_bias[i] += speculative_Label_Array[label].hidden_layer_1_bias_grad[i];
    for (int j = 0; j<n_inputs;j++){
      N_Network.hidden_weights_1[j][i] += speculative_Label_Array[label].hidden_weights_1_grad[j][i];
    }
  }
}

void update_Weights(int label){

  if(cross_entropy_loss(speculative_Label_Array[label].softmax_Outputs,N_Network.output_layer) > threshold){
    below_Thres_Flag = 1;
  }
  else{
    printf("Hi");
    //Applies Output Layer Gradients
    for (int i = 0; i<n_outputs;i++){
      N_Network.output_layer_bias[i] += speculative_Label_Array[label].output_layer_bias_grad[i];
      for (int j = 0; j<n_hidden_2;j++){
        N_Network.output_weights[j][i] += speculative_Label_Array[label].output_weights_grad[j][i];  
      }
    }
    //Applies Hidden Layer 2 Gradients
    for (int i = 0; i<n_hidden_2;i++){
      N_Network.hidden_layer_2_bias[i] += speculative_Label_Array[label].hidden_layer_2_bias_grad[i];
      for (int j = 0; j<n_hidden_1;j++){
        N_Network.hidden_weights_2[j][i] += speculative_Label_Array[label].hidden_weights_2_grad[j][i];
      }
    }
    //Applies Hidden Layer 1 Gradients
    for (int i = 0; i<n_hidden_1;i++){
      N_Network.hidden_layer_1_bias[i] += speculative_Label_Array[label].hidden_layer_1_bias_grad[i];
      for (int j = 0; j<n_inputs;j++){
        N_Network.hidden_weights_1[j][i] += speculative_Label_Array[label].hidden_weights_1_grad[j][i];
      }
    }
  }

  //Setting the Stored Activations equal to the Forward Pass Values
  for(int i = 0; i<n_inputs;i++){
    stored_Act.input_layer_ex[i] = N_Network.input_layer[i];
  }
  for(int i = 0; i<n_hidden_1;i++){
    stored_Act.hidden_layer_1_ex[i] = N_Network.hidden_layer_1[i];
  }
  for(int i = 0; i<n_hidden_2;i++){
    stored_Act.hidden_layer_2_ex[i] = N_Network.hidden_layer_2[i];
  }
  for(int i = 0; i<n_outputs;i++){
    stored_Act.output_layer_ex[i] = N_Network.output_layer[i];
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

void train(double images[], unsigned char labels[], int im_Num,int epochs){ 
  time_t start2, end2;
  for(int f = 0; f<epochs; f++){
    printf("\nEpoch: %d\n",f);
    start2 = clock();
    double desired[10];
    for(int i = 0; i < im_Num; i++){  
      //Perform Feed Forward
      //Perform Backpropagation
      //Next Cycle
      for(int j = 0; j < 28*28;j++){
        N_Network.input_layer[j] = images[j+i*28*28];
      }
      feed_Forward();
      if(activation_Type == 'R'){
        compute_Softmax_Output();
      }
      //Sets Desired for the Images
      for(int g = 0; g < n_outputs;g++){
        if(labels[i] == g){
          desired[g] = 1;
        }
        else{
          desired[g] = 0;
        }
      }
      backpropagation(desired,labels[i]);
    }
    end2 = clock();
    printf("\nElapsed Time: %f seconds\n",(double)(end2-start2)/CLOCKS_PER_SEC);
  }
  end2 = clock();
}

void s_train(double images[], unsigned char labels[], int im_Num,int epochs){ 
  time_t start2, end2;
  char first_pass = 'T';
  //Looping Through Epochs
  for(int f = 0; f<epochs; f++){
    printf("\nEpoch: %d\n",f);
    start2 = clock();
    double desired[10];
    //Looping Through Images
    for(int i = 0; i < im_Num; i++){
      //Applying Inputs for Forward Pass  
      for(int j = 0; j < 28*28;j++){
        N_Network.input_layer[j] = images[j+i*28*28];
      }
      //Setting the Designed for each Image
      for(int g = 0; g < n_outputs;g++){
        if(labels[i] == g){
          desired[g] = 1;
        }
        else{
          desired[g] = 0;
        }
      }
      int temp_Label = (int)labels[i];
      if(first_pass == 'T'){

        feed_Forward_spec();
        backpropagation2(desired,temp_Label);
        //Setting the Stored Activations equal to the Forward Pass Values
        for(int i = 0; i<n_inputs;i++){
          stored_Act.input_layer_ex[i] = N_Network.input_layer[i];
        }
        for(int i = 0; i<n_hidden_1;i++){
          stored_Act.hidden_layer_1_ex[i] = N_Network.hidden_layer_1[i];
        }
        for(int i = 0; i<n_hidden_2;i++){
          stored_Act.hidden_layer_2_ex[i] = N_Network.hidden_layer_2[i];
        }
        for(int i = 0; i<n_outputs;i++){
          stored_Act.output_layer_ex[i] = N_Network.output_layer[i];
        }
        first_pass == 'F';

      }
      else{
        #pragma omp parallel default(none) shared(desired,temp_Label)
        #pragma omp single 
        {

          #pragma omp task
          feed_Forward_spec();

          #pragma omp task
          backpropagation_spec(desired,temp_Label);

          #pragma omp taskwait
        }
      }


        if(below_Thres_Flag == 1){
          //Applies Output Layer Gradients
          for (int i = 0; i<n_outputs;i++){
            N_Network.output_layer_bias[i] += speculative_Label_Array[temp_Label].output_layer_bias_grad[i];
            for (int j = 0; j<n_hidden_2;j++){
              N_Network.output_weights[j][i] += speculative_Label_Array[temp_Label].output_weights_grad[j][i];  
            }
          }
          //Applies Hidden Layer 2 Gradients
          for (int i = 0; i<n_hidden_2;i++){
            N_Network.hidden_layer_2_bias[i] += speculative_Label_Array[temp_Label].hidden_layer_2_bias_grad[i];
            for (int j = 0; j<n_hidden_1;j++){
              N_Network.hidden_weights_2[j][i] += speculative_Label_Array[temp_Label].hidden_weights_2_grad[j][i];
            }
          }
          //Applies Hidden Layer 1 Gradients
          for (int i = 0; i<n_hidden_1;i++){
            N_Network.hidden_layer_1_bias[i] += speculative_Label_Array[temp_Label].hidden_layer_1_bias_grad[i];
            for (int j = 0; j<n_inputs;j++){
              N_Network.hidden_weights_1[j][i] += speculative_Label_Array[temp_Label].hidden_weights_1_grad[j][i];
            }
          }

          below_Thres_Flag = 0;
        }




      //Weight Update
      update_Weights(temp_Label);
    }
    end2 = clock();
    printf("\nElapsed Time: %f seconds\n",(double)(end2-start2)/CLOCKS_PER_SEC);
  }
}

void test(double images[], unsigned char labels[]){
  int max_Pos = 0;
  double count = 0;
  for(int k = 0;k<9999;k++){
    for(int j = 0; j < n_inputs;j++){
        N_Network.input_layer[j] = images[j+k*n_inputs];
      }
    feed_Forward();
    max_Pos = get_Max(N_Network.output_layer);
    if(max_Pos == labels[k]){
      count++; 
    }
  }
  compute_Softmax_Output();
  printf("\nCorrect Rate: %f",count/10000.0);
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
    //Variable Declaration
    unsigned char *training_Labels;                                               //Pointer for Training Labels Array (Space Allocated Later)
    unsigned char *testing_Labels;                                                //Pointer for Testing Labels Array (Space Allocated Later)
    double *training_Images;                                                      //Pointer for Training Images Array (Space Allocated Later)
    double *testing_Images;                                                       //Pointer for Testing Images Array (Space Allocated Later)

    //Random Initalization
    srand((unsigned int)time(NULL));                                              //Generate Random Seed for network initialization

    //Importing Data
    load_MNIST_Labels("train-labels.idx1-ubyte",&training_Labels);                //Loads Training Labels
    load_MNIST_Labels("t10k-labels.idx1-ubyte",&testing_Labels);                  //Loads Testing Labels
    load_MNIST_Images("train-images.idx3-ubyte",&training_Images);                //Loads Training Images
    load_MNIST_Images("t10k-images.idx3-ubyte",&testing_Images);                  //Loads Testing Images
    print_MNIST_Images(testing_Images,1);

    //Neural Network
    init_neural_network_Xavier();                                                 //Initializes Neural Network with Random Values

    int epochs = 9;
    time_t start, end;
    start = clock();
    if(speculative_Backpropagation == 'N'){
      train(training_Images,training_Labels,60000,epochs);
    }
    else if(speculative_Backpropagation == 'Y'){
      s_train(training_Images,training_Labels,60000,epochs);
    }
    end = clock();
    test(testing_Images,testing_Labels);
    printf("\nEpochs: %d\n",epochs);
    printf("\nElapsed Time: %f seconds\n",(double)(end-start)/CLOCKS_PER_SEC);

    //Deallocate Space 
    free(training_Labels); 
    free(testing_Labels); 
    free(training_Images);
    free(testing_Images);
    return 0;
}

