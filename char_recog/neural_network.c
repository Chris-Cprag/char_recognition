//Character Recognition in C
//-------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 

#define learning_rate 0.05

#define n_inputs 784 //# of Input Nodes
#define n_hidden_1 35     //37 for Sigmoid
#define n_hidden_2 35
#define n_outputs 10 //# of Output Nodes

//HL1 = Hidden Layer 1
//HL2 = Hidden Layer 2
typedef struct {
    double input_layer[n_inputs]; //Establishes the Input layer
    double hidden_layer_1[n_hidden_1]; //Establishes the HL1
    double hidden_layer_2[n_hidden_2]; //Establishes the HL2
    double output_layer[n_outputs]; //Establishes the Output layer
    
    double hidden_layer_1_bias[n_hidden_1]; //Establishes the Bias for the HL1
    double hidden_layer_2_bias[n_hidden_2]; //Establishes the Bias for the HL2
    double output_layer_bias[n_outputs]; //Establishes the Bias for the Output Layer
    
    double hidden_weights_1[n_inputs][n_hidden_1]; //Connection between the input and HL1
    double hidden_weights_2[n_hidden_1][n_hidden_2]; //Connection between the HL1 and HL2
    double output_weights[n_hidden_2][n_outputs]; //Connection between the HL2 and the output
} Neural_Network;

//Creating the Network
Neural_Network N_Network;
char activation_Type = 'S';  //Selection for Weight Activation Functions, R for Relu, S for Sigmoid
//-------------------------------------------------------
double sigmoid(double x){
  return 1/(1+exp(-x));
}

double sigmoid_D(double x){
  return sigmoid(x)*(1-sigmoid(x));
}


double re_lu(double x){
  if(x > 0){
    if(x > 1){
      return 1;
    }
    else{
      return x;
    }
  }
  else{
    return 0.001*x;
  }
}

double re_lu_D(double x){
  if(x <= 0){
    return 0.001;
  }
  else{
    if(x > 1){
      return 0.001;
    }
    else{
      return 1;
    }
  }
}


/*
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
*/

void compute_Softmax_Output(){ 
  double sum = 0.0;
  for(int i = 0; i < n_outputs; i++){
    sum += exp(N_Network.output_layer[i]);
  }

  for(int i = 0; i < n_outputs; i++){
    N_Network.output_layer[i] = exp(N_Network.output_layer[i])/sum;
  }
}

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
  printf("Address Location: %s has %d Labels.\n", address,image_Num);              //Prints the number of labels in the File

  int rows = 0;
  fread(&rows,sizeof(int),1,fp);                                                  //Extracts the number of rows
  rows = __builtin_bswap32(rows);  

  int columns = 0;
  fread(&columns,sizeof(int),1,fp);                                               //Extracts the number of columns
  columns = __builtin_bswap32(columns);  
  
  int image_Size = image_Num*rows*columns;

  unsigned char *temp = (unsigned char*)malloc(image_Size*sizeof(unsigned char)); //Allocates space for a temporary array that take in the bytes
  fread(temp,sizeof(unsigned char),image_Size,fp);                               //Reads the images into the array
  fclose(fp);

  *images = (double*)malloc(image_Size*sizeof(double));                           //Allocate the space for the Image Array in double form (0-1)

  for(int i = 0; i < image_Size; i++){                                             
    *(*images + i) = temp[i]/ 255.0;                                             //Fits the data between 0-1 and puts it into Image (Has to be decimal 255.0)
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
  for(int i = 0; i<28;i++){
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

void init_neural_network(){
  //Randomize Weights
  //Randomizes HD1 Weights
  for(int i = 0; i<n_inputs;i++){
    for(int j = 0; j<n_hidden_1; j++){
      N_Network.hidden_weights_1[i][j] = ((double)rand())/((double)RAND_MAX); //random number/max of random numbers
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

  //Randomize Bias
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

void backpropagation(double desired[])
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

void train(double images[], unsigned char labels[], int im_Num,int epochs){
  for(int f = 0; f<epochs; f++){
    double desired[10];
    for(int i = 0; i < im_Num; i++){  
      //Perform Feed Forward
      //Perform Backpropagation
      //Next Cycle
      for(int j = 0; j < 28*28;j++){
        N_Network.input_layer[j] = images[j+i*28*28];
      }
      feed_Forward();
      printf("\n%f ",N_Network.output_layer[0]);
      printf("%f ",N_Network.output_layer[1]);
      printf("%f ",N_Network.output_layer[2]);
      printf("%f ",N_Network.output_layer[3]);
      printf("%f ",N_Network.output_layer[4]);
      printf("%f ",N_Network.output_layer[5]);
      printf("%f ",N_Network.output_layer[6]);
      printf("%f ",N_Network.output_layer[7]);
      printf("%f ",N_Network.output_layer[8]);
      printf("%f\n",N_Network.output_layer[9]);
      //compute_Softmax_Output();
      if(labels[i] == 0){
        desired[0] = 1;
        desired[1] = 0;
        desired[2] = 0;
        desired[3] = 0;
        desired[4] = 0;
        desired[5] = 0;
        desired[6] = 0;
        desired[7] = 0;
        desired[8] = 0;
        desired[9] = 0;
        backpropagation(desired);
      }
      else if(labels[i] == 1){
        desired[0] = 0;
        desired[1] = 1;
        desired[2] = 0;
        desired[3] = 0;
        desired[4] = 0;
        desired[5] = 0;
        desired[6] = 0;
        desired[7] = 0;
        desired[8] = 0;
        desired[9] = 0;
        backpropagation(desired);
      }
      else if(labels[i] == 2){
        desired[0] = 0;
        desired[1] = 0;
        desired[2] = 1;
        desired[3] = 0;
        desired[4] = 0;
        desired[5] = 0;
        desired[6] = 0;
        desired[7] = 0;
        desired[8] = 0;
        desired[9] = 0;
        backpropagation(desired);
      }
      else if(labels[i] == 3){
        desired[0] = 0;
        desired[1] = 0;
        desired[2] = 0;
        desired[3] = 1;
        desired[4] = 0;
        desired[5] = 0;
        desired[6] = 0;
        desired[7] = 0;
        desired[8] = 0;
        desired[9] = 0;
        backpropagation(desired);
      }
      else if(labels[i] == 4){
        desired[0] = 0;
        desired[1] = 0;
        desired[2] = 0;
        desired[3] = 0;
        desired[4] = 1;
        desired[5] = 0;
        desired[6] = 0;
        desired[7] = 0;
        desired[8] = 0;
        desired[9] = 0;
        backpropagation(desired);
      }
      else if(labels[i] == 5){
        desired[0] = 0;
        desired[1] = 0;
        desired[2] = 0;
        desired[3] = 0;
        desired[4] = 0;
        desired[5] = 1;
        desired[6] = 0;
        desired[7] = 0;
        desired[8] = 0;
        desired[9] = 0;
        backpropagation(desired);
      }
      else if(labels[i] == 6){
        desired[0] = 0;
        desired[1] = 0;
        desired[2] = 0;
        desired[3] = 0;
        desired[4] = 0;
        desired[5] = 0;
        desired[6] = 1;
        desired[7] = 0;
        desired[8] = 0;
        desired[9] = 0;
        backpropagation(desired);
      }
      else if(labels[i] == 7){
        desired[0] = 0;
        desired[1] = 0;
        desired[2] = 0;
        desired[3] = 0;
        desired[4] = 0;
        desired[5] = 0;
        desired[6] = 0;
        desired[7] = 1;
        desired[8] = 0;
        desired[9] = 0;
        backpropagation(desired);
      }
      else if(labels[i] == 8){
        desired[0] = 0;
        desired[1] = 0;
        desired[2] = 0;
        desired[3] = 0;
        desired[4] = 0;
        desired[5] = 0;
        desired[6] = 0;
        desired[7] = 0;
        desired[8] = 1;
        desired[9] = 0;
        backpropagation(desired);
      }
      else if(labels[i] == 9){
        desired[0] = 0;
        desired[1] = 0;
        desired[2] = 0;
        desired[3] = 0;
        desired[4] = 0;
        desired[5] = 0;
        desired[6] = 0;
        desired[7] = 0;
        desired[8] = 0;
        desired[9] = 1;
        backpropagation(desired);
      }
      else{
        printf("Out of Bounds Input");
      }
    }
  }
}

int get_Max(double arr[]){
  double max = arr[0];
  int enume = 0;
  for(int i = 1; i < n_outputs; i++){
    if(max < arr[i]){
      max = arr[i];
      enume = i;
    }
  }
  return enume;
}

void test(double images[], unsigned char labels[]){
  int max_Pos = 0;
  double count = 0;
  for(int k = 0;k<9999;k++){
    for(int j = 0; j < n_inputs;j++){
        N_Network.input_layer[j] = images[j+k*n_inputs];
      }
    feed_Forward();
    
    compute_Softmax_Output();
    max_Pos = get_Max(N_Network.output_layer);
    if(max_Pos == labels[k]){
      count++; 
    }
  }
  printf("\nCorrect Rate: %f",count/10000.0);
}


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
    for(int i = 0;i<10;i++){
      printf("%d\n",training_Labels[i]);
    }

    load_MNIST_Labels("t10k-labels.idx1-ubyte",&testing_Labels);                  //Loads Testing Labels
    for(int i = 0;i<10;i++){
      printf("%d\n",testing_Labels[i]);
    }

    load_MNIST_Images("train-images.idx3-ubyte",&training_Images);                //Loads Training Images

    load_MNIST_Images("t10k-images.idx3-ubyte",&testing_Images);                  //Loads Testing Images

    print_MNIST_Images(testing_Images,1);

    //Neural Network
    init_neural_network();                                                        //Initializes Neural Network with Random Values

    int epochs = 9;
    time_t start, end;
    start = clock();
    train(training_Images,training_Labels,60000,epochs);
    test(testing_Images,testing_Labels);
    end = clock();
    printf("\nEpochs: %d\n",epochs);
    printf("\nElapsed Time: %f seconds\n",(double)(end-start)/CLOCKS_PER_SEC);

    //Deallocate Space 
    free(training_Labels); 
    free(testing_Labels); 
    free(training_Images);
    free(testing_Images);
    return 0;
}

//Currently having trouble with the ReLu function
