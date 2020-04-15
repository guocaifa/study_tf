import numpy
import scipy.special
import scipy.misc
import matplotlib.pyplot

# neural network class definition\n",
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # link weight matrices, wih and who\n",
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer\n",
        # w11 w21\n",
        # w12 w22 etc \n",
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        # learning rate\n",
        self.lr = learningrate

        # activation function is the sigmoid function\n",
        self.activation_function = lambda x: scipy.special.expit(x)


    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer\n",
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))


    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

if __name__ == "__main__":
    inputNodes   = 784
    hiddenNodes  = 100
    outputNodes  = 10
    learningRate = 0.3
    epochs       = 4  #测试几次世

    n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

    trainingFile = open("mnist_dataset/mnist_train_100.csv", 'r')
    trainingList = trainingFile.readlines()
    trainingFile.close()

    for e in range(epochs):
        for record in trainingList:
            allValues = record.split(',')
            inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01

            targets = numpy.zeros(outputNodes) + 0.01
            targets[int(allValues[0])] = 0.99

            n.train(inputs, targets)

    testFile = open("mnist_dataset/mnist_test_10.csv", 'r')
    testList = testFile.readlines()
    testFile.close()

    imageFilePath = "my_own_images/2828_my_own_2.png"

    imgArray = scipy.misc.imread(imageFilePath, flatten = True)
    imgData  = 255.0 - imgArray.reshape(784)
    imgData  = (imgData / 255.0 * 0.99) + 0.01

    outputNum = n.query(imgData)
    print(outputNum)
    print(numpy.argmax(outputNum))

    scorecard = []

    for record in testList:
        # split the record by the ',' commas\n",
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs\n",
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

    scorecardArray = numpy.asarray(scorecard)
    print("performance = ",scorecardArray.sum() / scorecardArray.size)
