def elem_add_backward(grad):
  return [grad, grad]
def scal_add_backward(grad):
  return [grad]
def elem_sub_backward(grad):
  return [grad,  grad * -1]
def scal_sub_backward(grad):
  return [grad]
def elem_mul_backward(grad, x, y):
  return [y * grad, x * grad]
def scal_mul_backward(grad, c):
  return [grad * c]
def elem_div_backward(grad, x, y):
  return [y**-1 * grad, x * -1 * y**-2 * grad]
def scale_div_backward(grad, c):
  return [grad / c]
def trans_backward(grad, x):
  return [x.T @ grad]