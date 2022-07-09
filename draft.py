# -*- coding: utf-8 -*
#!/usr/bin/env python

import numpy  as np
import xlrd
import xlsxwriter
import random
'''
a = np.random.randint(0,10,size=(5,4,8))
print('a=',a)
b= a[0,:,:]
a[0,:,:] = np.delete(a[0,:,:],[0,1],axis = 0)

print('after a=',a)

for i in range(3):
    locals()['a'+str(i)]=np.zeros((i,i))
    print('a'+str(i))
    print(eval('a'+str(i)))


names = locals()
for i in range(2,4):
    names['n' + str(i)] = np.zeros((i,i))
    names['n' + str(i)] = np.delete(names['n' + str(i)],0,axis = 0)
for i in range(2,4):
    print(names.get('n' + str(i)))
'''
data = np.random.rand(4,8)
data[3,5] = float('nan')
data[np.isnan(data)]=0
print(data)


#print(list1[0][2:4])
#print(np.array(list1[0][0])+np.array(list[0][1]))
