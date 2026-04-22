"""Single neuron with 3 inputs, weights and a bias"""
inputs = [1,2,3]
weights = [0.2,0.4,0.6]
bias = 2

output = (inputs[0]*weights[0]+
          inputs[1]*weights[1]+
          inputs[2]*weights[2]+
          bias)

print(output)


"""A Layer of Neurons """

inputs = [1,2,3,2.5]
weights1 = [0.2, 0.4, 0.6, 0.5]
weights2 = [0.4, -0.5, 0.6, 0.5]
weights3 = [-0.3, 0.7, 0.1, -0.8]
bias1 = 2
bias2 = 3
bias3 = 0.5


output = (# Neuron 1
          inputs[0]*weights1[0]+
          inputs[1]*weights1[1]+
          inputs[2]*weights1[2]+
          inputs[3]*weights1[3]+
          bias1,
          # Neuron 2
          inputs[0]*weights2[0]+
          inputs[1]*weights2[1]+
          inputs[2]*weights2[2]+
          inputs[3]*weights2[3]+
          bias2,
          # Neuron 3
          inputs[0]*weights3[0]+
          inputs[1]*weights3[1]+
          inputs[2]*weights3[2]+
          inputs[3]*weights3[3]+
          bias3)
print(output)