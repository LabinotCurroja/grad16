



class MatmulGradOp:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, grad_output):
        """Calculate the gradients for matmul."""
        # Calculate gradients for each operand
        grad_a = grad_output * self.b.cpu_data.T()
        grad_b = self.a.cpu_data.T() * grad_output

        # Return the gradients
        return grad_a, grad_b