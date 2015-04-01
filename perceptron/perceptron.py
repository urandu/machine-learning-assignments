

threshold = 0.5
learning_rate = 0.1
weights = [0, 0, 0]
training_set = [((1, 0, 0), 1), ((1, 0, 1), 1), ((1, 1, 0), 1), ((1, 1, 1), 0)]
 
def dot_product(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))
 
while True:
    #print the seperation line
    print('-' * 60)
    error_count = 0
    for input_vector, desired_output in training_set:
	#printing the initial weights
        print("Initial Weights: {}". format(weights))
        result = dot_product(input_vector, weights) > threshold
	#calculating error
        error = desired_output - result
	print("Error: {}". format(error))
	#if there is an error adjust the weights
        if error != 0:
            error_count += 1
	    #for every input in the input vector calculate the new weights
            for index, value in enumerate(input_vector):
                weights[index] += learning_rate * error * value
	    print("Final Weights: {}". format(weights))
    #if there was no input with an error terminate the loop
    if error_count == 0:
        break


