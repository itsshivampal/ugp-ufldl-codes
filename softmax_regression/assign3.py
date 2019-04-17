import numpy as np
import scipy.optimize

def load_MNIST_images(filename):
    with open(filename, "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        n_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((int(n_images), int(rows * cols)))
        images = images.T
        images = images.astype(np.float64) / 255
        f.close()
    return images


def load_MNIST_labels(filename):
    with open(filename, 'r') as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        n_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        labels = np.fromfile(f, dtype=np.uint8)
        f.close()
    return labels

    
def compute_numerical_gradient(J, theta):
    n = theta.size
    grad = np.zeros(n)
    eps = 1.0e-4
    eps2 = 2*eps
    for i in range(n):
        theta_p = theta.copy()
        theta_n = theta.copy()
        theta_p[i] = theta[i] + eps
        theta_n[i] = theta[i] - eps
        grad[i] = (J(theta_p) - J(theta_n)) / eps2
    return grad



def softmax_cost(theta, n_classes, input_size, lambda_, data, labels):
    k = n_classes
    n, m = data.shape
    
    theta = theta.reshape((k, n))

    theta_data = theta.dot(data)
    alpha = np.max(theta_data, axis=0)
    theta_data -= alpha
    proba = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
    
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.arange(m))))
    indicator = np.array(indicator.todense())
    
    cost = -1.0/m * np.sum(indicator * np.log(proba)) + 0.5*lambda_*np.sum(theta*theta)
    
    grad = -1.0/m * (indicator - proba).dot(data.T) + lambda_*theta
    
    grad = grad.ravel()
    
    return cost, grad


def softmax_train(input_size, n_classes, lambda_, input_data, labels, options={'maxiter': 400, 'disp': True}):
    theta = 0.005 * np.random.randn(n_classes * input_size)
    J = lambda theta : softmax_cost(theta, n_classes, input_size, lambda_, input_data, labels)
    results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
    opt_theta = results['x']
    model = {'opt_theta': opt_theta, 'n_classes': n_classes, 'input_size': input_size}
    return model

def softmax_predict(model, data):
    theta = model['opt_theta']
    k = model['n_classes']
    n = model['input_size']
    theta = theta.reshape((k, n))
    theta_data = theta.dot(data)
    alpha = np.max(theta_data, axis=0)
    theta_data -= alpha
    proba = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)

    pred = np.argmax(proba, axis=0)

    return pred



input_size = 28 * 28
n_classes = 10
lambda_ = 1e-4


images = load_MNIST_images('train-images-idx3-ubyte')
labels = load_MNIST_labels('train-labels-idx1-ubyte')
input_data = images

# Randomly initialise theta
theta = 0.005 * np.random.randn(n_classes * input_size)

cost, grad = softmax_cost(theta, n_classes, input_size, lambda_, input_data, labels)



debug = False
if debug:
    J = lambda theta : softmax_cost(theta, n_classes, input_size, lambda_, input_data, labels)[0]
    numgrad = compute_numerical_gradient(J, theta)

    n = min(grad.size, 20)
    for i in range(n):
        print("{0:20.12f} {1:20.12f}".format(numgrad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print("Norm of difference = ", diff)


options = {'maxiter': 100, 'disp': True}
model = softmax_train(input_size, n_classes, lambda_, input_data, labels, options)


images = load_MNIST_images('train-images-idx3-ubyte')
labels = load_MNIST_labels('train-labels-idx1-ubyte')
input_data = images

pred = softmax_predict(model, input_data)

acc = np.mean(labels == pred)
print("Accuracy: {:5.2f}% \n".format(acc*100))
