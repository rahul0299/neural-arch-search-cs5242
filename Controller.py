# VAIBHAV
#
# TODO: create a vocabulary (For now manual, in future try to be dynamic using a generator)
# NOTE - for simple MLP number range is fine
# TODO: RNN to generate tuples
# [conv, 3x3, 64], [conv, 5x5, 128], [pool, max, 2x2], [dense, 256], [output, softmax, 10]
# start with "[START]" token and finish model with "[END]" token

# CURRENT PLAN
# Generate small model (upto 5 layers) of MLP type.
# For each layer only need to predict the number of neurons in hidden layer (1 hyperparameter)
# Start with bounds 1-10 ie our vocab
# Generate a probability distribution over 1-10 of the next layer containing i neurons [1 <= i <= 10]
# Start generating the sequence by forward pass through RNN
# Use the generated sequence to build and train child network
# Use RL to update RNN weights -- NEXT WEEK (20th Mar+)



### Network Output ->   [["START"], ..... upto 5 numbers, ["END"]]
#### Example1  ->  [["START"], [5], [2], [8], [7], [6], ["END"]]
#### Example2 ->  [["START"], [4], [3], [8], ["END"]]
