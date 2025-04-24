//Character Recognition in C
//-------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h> 
#include <windows.h>

//Constant Declaration
#define learning_rate 0.01

#define n_inputs 784                                    //# of Input Nodes
#define n_hidden_1 35                                   //# of Hidden Nodes 
#define n_hidden_2 35                                   //...
#define n_outputs 10                                    //# of Output Nodes

#define BATCH_SIZE 0                                  //Size per Batch

//(2 Hidden Layer) Neural Network Structure
typedef struct {
  double input_layer[n_inputs];                            //Activations
  double hidden_layer_1[n_hidden_1];                       //,,,
  double hidden_layer_2[n_hidden_2];                       //...
  double output_layer[n_outputs];                          //...
} activations;

typedef struct {
    activations act[2];                                      //Double Ended Activations
    
    double hidden_layer_1_bias[n_hidden_1];                  //Bias
    double hidden_layer_2_bias[n_hidden_2];                  //...
    double output_layer_bias[n_outputs];                     //...
    
    double hidden_weights_1[n_inputs][n_hidden_1];           //Weights
    double hidden_weights_2[n_hidden_1][n_hidden_2];         //...
    double output_weights[n_hidden_2][n_outputs];            //...

    double correct_out[n_outputs];                           //Previous Desired
    int correct_label;
} Neural_Network;

typedef struct {
  double hidden_weights_1_grad[n_inputs][n_hidden_1];      //Weights Gradients
  double hidden_weights_2_grad[n_hidden_1][n_hidden_2];    //...
  double output_weights_grad[n_hidden_2][n_outputs];       //...

  double hidden_layer_1_bias_grad[n_hidden_1];             //Bias Gradients
  double hidden_layer_2_bias_grad[n_hidden_2];             //...
  double output_layer_bias_grad[n_outputs];                //...

} label_Grad;

//Global Initalization
Neural_Network N_Network; //Neural Network Data Structure
label_Grad str_gradients; //Store Gradients from the Backpropagation
label_Grad gradients_labels[n_outputs]; 
double perc_distr[n_outputs][n_outputs];
int circ = 0; //Circular Value

//Settings
char activation_Type = 'R';                              //Activation Type: S for Sigmoid, R for ReLu, L for Leaky ReLu
char speculative_Backpropagation = 'Y';                  //Speculative Backpropagation? Y for Yes, N for No
float threshold = 0.25;
int batch_count = 0; 

//Measurements
LARGE_INTEGER freq, start, end;
double durationInSeconds = 0; 

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
    sum += exp(N_Network.act[circ].output_layer[i]);
  }

  for(int j = 0; j < n_outputs; j++){
    N_Network.act[circ].output_layer[j] = exp(N_Network.act[circ].output_layer[j])/sum;
  }
}

double cross_entrpy(double arr1[],double arr2[]){
  double sum = 0; 
  for(int i = 0; i <n_outputs;i++){
    sum += arr1[i]*log(arr2[i]+1e-15); //Arr2 is predicted
  }
  return -sum;
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
      sum+=N_Network.act[circ].input_layer[j]*N_Network.hidden_weights_1[j][i];
    }
    if(activation_Type == 'R'){
      N_Network.act[circ].hidden_layer_1[i] = re_lu(N_Network.hidden_layer_1_bias[i] + sum);
    }
    else if(activation_Type == 'S'){
      N_Network.act[circ].hidden_layer_1[i] = sigmoid(N_Network.hidden_layer_1_bias[i] + sum);
    }
  }

  //SUM GREATER THAN 1 BECAUSE A BUNCH OF DECIMALS MULTIPLING TOGETHER
  //ACT(   2.94 FOR RELU would be 2.94 ACT ON THE NEXT PASS IT GROWS EXPONENTIALLY )

  //Get HDL2 Activation
  for(int i = 0; i<n_hidden_2;i++){
    double sum = 0;
    for(int j = 0;j<n_hidden_1;j++){
      sum+=N_Network.act[circ].hidden_layer_1[j]*N_Network.hidden_weights_2[j][i];
    }
    if(activation_Type == 'R'){
      N_Network.act[circ].hidden_layer_2[i] = re_lu(N_Network.hidden_layer_2_bias[i] + sum);
    }
    else if(activation_Type == 'S'){
      N_Network.act[circ].hidden_layer_2[i] = sigmoid(N_Network.hidden_layer_2_bias[i] + sum);
    }
  }

  //Get Output Layer Activation
  for(int i = 0; i<n_outputs;i++){
    double sum = 0;
    for(int j = 0;j<n_hidden_2;j++){
      sum+=N_Network.act[circ].hidden_layer_2[j]*N_Network.output_weights[j][i];
    }
    if(activation_Type == 'R'){
      N_Network.act[circ].output_layer[i] = re_lu(N_Network.output_layer_bias[i] + sum);
    }
    else if(activation_Type == 'S'){
      N_Network.act[circ].output_layer[i] = sigmoid(N_Network.output_layer_bias[i] + sum);
    }
  }

  //SoftMax for ReLu
  if(activation_Type == 'R'){
    compute_Softmax_Output();
  }

}

void backpropagation()
{
  //Calculating Error Deltas
  //Gives you the Output Error Delta
  double o_error[n_outputs];
  for(int i = 0;i<n_outputs;i++){
    if(activation_Type == 'R'){
      o_error[i]=(N_Network.correct_out[i]-N_Network.act[!circ].output_layer[i])*re_lu_D(N_Network.act[!circ].output_layer[i]);
    }
    else if(activation_Type == 'S'){
      o_error[i]=(N_Network.correct_out[i]-N_Network.act[!circ].output_layer[i])*sigmoid_D(N_Network.act[!circ].output_layer[i]);
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
      H2_error[i] = sum*re_lu_D(N_Network.act[!circ].hidden_layer_2[i]); //Finds the error delta after the sum
    }
    else if(activation_Type == 'S'){
      H2_error[i] = sum*sigmoid_D(N_Network.act[!circ].hidden_layer_2[i]); //Finds the error delta after the sum
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
      H1_error[i] = sum*re_lu_D(N_Network.act[!circ].hidden_layer_1[i]); 
    }
    else if(activation_Type == 'S'){
      H1_error[i] = sum*sigmoid_D(N_Network.act[!circ].hidden_layer_1[i]); 
    }
  }
  
  //Calculating the Adjusted Weights and Bias
  //Adjusts the Hidden_2-Output Weights and Bias
  //printf("BackProp %d: \n",backup.correct_label);
  for (int i = 0; i<n_outputs;i++){
    gradients_labels[N_Network.correct_label].output_layer_bias_grad[i] = o_error[i]*learning_rate;
    for (int j = 0; j<n_hidden_2;j++){
      gradients_labels[N_Network.correct_label].output_weights_grad[j][i] =  learning_rate*o_error[i]*N_Network.act[!circ].hidden_layer_2[j];
    }
  }
  //Adjusts the Hidden_1-Hidden_2 Weights and Bias
  for (int i = 0; i<n_hidden_2;i++){
    gradients_labels[N_Network.correct_label].hidden_layer_2_bias_grad[i] = H2_error[i]*learning_rate;
    for (int j = 0; j<n_hidden_1;j++){
      gradients_labels[N_Network.correct_label].hidden_weights_2_grad[j][i] = learning_rate*H2_error[i]*N_Network.act[!circ].hidden_layer_1[j];
    }
  }
  //Adjusts the Input-Hidden_1 Weights and Bias
  for (int i = 0; i<n_hidden_1;i++){
    gradients_labels[N_Network.correct_label].hidden_layer_1_bias_grad[i] = H1_error[i]*learning_rate;
    for (int j = 0; j<n_inputs;j++){
      gradients_labels[N_Network.correct_label].hidden_weights_1_grad[j][i] = learning_rate*H1_error[i]*N_Network.act[!circ].input_layer[j];
    }
  }
  
}

void weight_updates(){
  //Adjusts Weights
  //printf("Weight Update %d: \n",backup.correct_label);
  //Applies the sum of error
  for (int i = 0; i<n_outputs;i++){
    N_Network.output_layer_bias[i] += str_gradients.output_layer_bias_grad[i];
    str_gradients.output_layer_bias_grad[i] = 0; 
    for (int j = 0; j<n_hidden_2;j++){
      N_Network.output_weights[j][i] += str_gradients.output_weights_grad[j][i];  
      str_gradients.output_weights_grad[j][i] = 0; 
    }
  }
  //Applies Hidden Layer 2 Gradients
  for (int i = 0; i<n_hidden_2;i++){
    N_Network.hidden_layer_2_bias[i] += str_gradients.hidden_layer_2_bias_grad[i];
    str_gradients.hidden_layer_2_bias_grad[i] = 0; 
    for (int j = 0; j<n_hidden_1;j++){
      N_Network.hidden_weights_2[j][i] += str_gradients.hidden_weights_2_grad[j][i];
      str_gradients.hidden_weights_2_grad[j][i] = 0; 
    }
  }
  //Applies Hidden Layer 1 Gradients
  for (int i = 0; i<n_hidden_1;i++){
    N_Network.hidden_layer_1_bias[i] += str_gradients.hidden_layer_1_bias_grad[i];
    str_gradients.hidden_layer_1_bias_grad[i] = 0; 
    for (int j = 0; j<n_inputs;j++){
      N_Network.hidden_weights_1[j][i] += str_gradients.hidden_weights_1_grad[j][i];
      str_gradients.hidden_weights_1_grad[j][i] = 0; 
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

void test(double images[], unsigned char labels[]){
  int max_Pos = 0;
  double count = 0;
  for(int k = 0;k<9999;k++){
    for(int j = 0; j < n_inputs;j++){
        N_Network.act[circ].input_layer[j] = images[j+k*n_inputs];
      }
    feed_Forward();
    max_Pos = get_Max(N_Network.act[circ].output_layer);
    if(max_Pos == labels[k]){
      count++; 
    }
  }
  compute_Softmax_Output();
  printf("\nCorrect Rate: %f",count/10000.0);
  printf("\n");
  for(int i = 0;i<n_outputs;i++){
    printf("%f ",N_Network.act[circ].output_layer[i]);
  }
}

void train(double images[], unsigned char labels[], int im_Num,int epochs,double t_images[], unsigned char t_labels[]){ 
  time_t start2, end2;
  double desired[n_outputs];
  //Initial Pass -----
  for(int j = 0; j < n_inputs;j++){           //Sets Input
    N_Network.act[0].input_layer[j] = images[j];       
  }
  for(int g = 0; g < n_outputs;g++){          //Sets Output Label
    if(labels[0] == g){
      desired[g] = 1;
    }
    else{
      desired[g] = 0;
    }
  }
  feed_Forward();




  //Looped Passes -----
  for(int f = 0; f<epochs; f++){
    printf("\nEpoch: %d\n",f);
    start2 = clock();
    for(int i = 0; i < im_Num; i++){  
      circ = !circ;
      for(int g = 0; g < n_outputs;g++){
        if(labels[i] == g){
          desired[g] = 1;
        }
        else{
          desired[g] = 0;
        }
      }

      QueryPerformanceCounter(&start);

      //printf("Feed Forward %d: \n",labels[i]);
      #pragma omp parallel num_threads(2)
      {
        #pragma omp single nowait
        {
          for(int j = 0; j < n_inputs;j++){
            N_Network.act[circ].input_layer[j] = images[j+i*n_inputs];
          }
          feed_Forward();
        }
        #pragma omp single
        {
          //printf("%f\n",cross_entrpy(perc_distr[backup.correct_label],backup.output_layer_ex));
          if(speculative_Backpropagation == 'Y' && cross_entrpy(perc_distr[N_Network.correct_label],N_Network.act[!circ].output_layer) >= threshold)  //If The Difference between the Previous Outputs and Perc Distr @ Label is small enough
          {
            backpropagation();
          }
          else if(speculative_Backpropagation == 'N'){
            backpropagation();
          }
            //Add Previous Gradients to Summation
            for (int i = 0; i<n_outputs;i++){
              str_gradients.output_layer_bias_grad[i] += gradients_labels[N_Network.correct_label].output_layer_bias_grad[i];
              for (int j = 0; j<n_hidden_2;j++){
                str_gradients.output_weights_grad[j][i] += gradients_labels[N_Network.correct_label].output_weights_grad[j][i];  
              }
            }
            //Applies Hidden Layer 2 Gradients
            for (int i = 0; i<n_hidden_2;i++){
              str_gradients.hidden_layer_2_bias_grad[i] += gradients_labels[N_Network.correct_label].hidden_layer_2_bias_grad[i];
              for (int j = 0; j<n_hidden_1;j++){
                str_gradients.hidden_weights_2_grad[j][i] += gradients_labels[N_Network.correct_label].hidden_weights_2_grad[j][i];
              }
            }
            //Applies Hidden Layer 1 Gradients
            for (int i = 0; i<n_hidden_1;i++){
              str_gradients.hidden_layer_1_bias_grad[i] += gradients_labels[N_Network.correct_label].hidden_layer_1_bias_grad[i];
              for (int j = 0; j<n_inputs;j++){
                str_gradients.hidden_weights_1_grad[j][i] += gradients_labels[N_Network.correct_label].hidden_weights_1_grad[j][i];
              }
            }
            for(int i = 0; i<n_outputs;i++){
              perc_distr[N_Network.correct_label][i] = N_Network.act[!circ].output_layer[i];
            } 

            for(int i = 0; i<n_outputs;i++){
              N_Network.correct_out[i] = desired[i];
            }
            N_Network.correct_label = labels[i]; 
        }
      }
      QueryPerformanceCounter(&end);
      durationInSeconds = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart + durationInSeconds;
      //Apply Gradients after a certain number of summations
      if(batch_count >= BATCH_SIZE){
        weight_updates();
        batch_count = -1;
      }
      batch_count++; 
      
    }
    
    end2 = clock();
    //printf("\nElapsed Time: %f seconds\n",(double)(end2-start2)/CLOCKS_PER_SEC);
    test(t_images,t_labels);
    //printf("\nPer Step = %f ",durationInSeconds/60000);
    //durationInSeconds = 0; 
  }
  end2 = clock();
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
    QueryPerformanceFrequency(&freq);
    start = clock();
    train(training_Images,training_Labels,60000,epochs,testing_Images,testing_Labels);

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

