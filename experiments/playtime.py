import pandas as pd 
import numpy as np 


test1 = pd.DataFrame([1,2,3,4])
test2 = pd.DataFrame([5,6,7,8])

test3 = pd.concat([test1,test2], axis=1)
test3.columns = ['a','b']
print(test3)


