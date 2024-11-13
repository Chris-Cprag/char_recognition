# char_recognition
Measurements ------------------------------
Time is measured using the Time.h library in C, taking the time after training and subtracting it by the time before training.

The Rate of Correctness is measured by taking the counted number of correct identifications and dividing it by the total number of testing data

Other Information -------------------------------
ReLu and SoftMax functions have been fixed and fully implemented. ReLu+SoftMax can get an accuracy of 96% on average and Sigmoid can now get up to 86% accuracy.
The next step will be to try and implement Speculative Backpropagation in the next version.
