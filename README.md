# char_recognition
Measurements ------------------------------
Time is measured using the Time.h library in C, taking the time after training and subtracting it by the time before training.

The Rate of Correctness is measured by taking the counted number of correct identifications and dividing it by the total number of testing data

Other Information -------------------------------
As of right now, the Re Lu function has broken the code, where initally sigmoid was working even with the option to use ReLu, a revaluation fo the code is needed.
When sigmoid was working, the ReLu function was producing a constant correctness rate of 9.8% which when using a printf statement to check, showed that all the output
layer values were being set to 0, causing the final result to always be zero (Which gives a 9.8% correctness result)
  Multiple issues relating to this will be explored in the future, likely going to restart with a clean slate and delve into the theory of the program once again
  Sigmoid may not be working anymore due to a change that was made to the network that wasn't changed back after testing, once that is fixed it should work again
  seperately from ReLu.
  SoftMax should work but at the moment cannot be tested due to ReLu's faults
