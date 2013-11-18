
import sys
sys.path.append('../liblinear_python')

from liblinear import *
from  liblinearutil import *


print "this is liblinear test project"
y, x = [1,-1], [[1,0,1], [-1,0,-1]]
y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
prob  = problem(y, x)
param = parameter('-c 4 -B 1')
m = train(prob, param)
save_model('../data/heart_scale.model', m)
m = load_model('../data/heart_scale.model')
p_label, p_acc, p_val = predict(y, x, m, '-b 0')
ACC, MSE, SCC = evaluations(y, p_label)
print ACC, MSE, SCC
