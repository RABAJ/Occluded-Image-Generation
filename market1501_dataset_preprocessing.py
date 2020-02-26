#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import shutil

a=['0129','1451','0084']

se=set()
sed={}
i=0
for f_name in os.listdir('/home/ws2/Downloads/Market-1501/bounding_box_train'):
    if (f_name[0:4]) not in a:
        
        #print(int(f_name[0:4]))
        if f_name[0:4] not in se:
            i=i+1
            se.add(f_name[0:4])
            #if (int(f_name[0:4])) not in sed:
            sed[(f_name[0:4])]=0
                #print(sed[int(f_name[0:4])])
            shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_train/'+f_name, '/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+str((f_name[0:4]))+'_'+str(sed[(f_name[0:4])])+'.jpg')
        else:
            i=i+1
            #print(i)
            sed[(f_name[0:4])]=sed[(f_name[0:4])]+1
            shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_train/'+f_name, '/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+str((f_name[0:4]))+'_'+str(sed[(f_name[0:4])])+'.jpg')
    else:
        print(int(f_name[0:4]))
            #shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_train/'+f_name, '/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+str((f_name[0:4]))+'_'+str(sed[(f_name[0:4])])+'.jpg')
print(i)


# In[25]:


import os

import shutil

se1=set()

sed1={}

for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1'):

    if f_name[0:4] not in se1:

        se1.add(f_name[0:4])
        sed1[f_name[0:4]]=1

        #print(int(f_name[0:4]))

    else:

        #print(f_name)

        

        sed1[f_name[0:4]]=sed1[f_name[0:4]]+1

#print(sed1)
sed2={}

i=0
print(len(sed1))
s=0
for key,value in sed1.items():
    s=s+value
    
print('s',s)
for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1'):

    i=i+1

    so=int(f_name[5:-4])

    s1=sed1[f_name[0:4]]
    #print(f_name[0:4],s1)
    #print("i",i)

    #print(s1)

    #sed2[f_name]=str(f_name[0:5])+str((so+1)%sed1[f_name[0:4]])+'.jpg'

    shutil.copy('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+f_name, '/home/ws2/Downloads/Market1501_train_test/nonocc_train_diffpose_v1/'+str(f_name[0:5])+str((so+1)%sed1[f_name[0:4]])+'.jpg')
    #shutil.copy('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+f_name, '/home/ws2/Downloads/Market1501_train_test/ave/'+f_name[:-4]+'.jpg')
print(s1)

#print(len(se1))

#print(len(sed1))


# In[28]:


p=list(se1)
aa={}
i=0
for x in p:
    aa[x]=i
    i=i+1
    


# In[29]:


print(len(p))


# In[31]:


for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1'):
    inn=aa[f_name[0:4]]
    ad=int(f_name[5:-4])
    pp=p[(inn+ad+1)%748]
    shutil.copy('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+pp+'_'+str(0)+'.jpg', '/home/ws2/Downloads/Market1501_train_test/nonocc_train_diffiden_v1/'+str(f_name[:-4])+'.jpg')
    
Jupyter Notebook
Untitled15 Last Checkpoint: 3 hours ago (autosaved) Current Kernel Logo 

Python 3

    File
    Edit
    View
    Insert
    Cell
    Kernel
    Widgets
    Help

import os

import shutil

​

a=['0129','1451','0084']

​

se=set()

sed={}

i=0

for f_name in os.listdir('/home/ws2/Downloads/Market-1501/bounding_box_train'):

    if (f_name[0:4]) not in a:

        

        #print(int(f_name[0:4]))

        if f_name[0:4] not in se:

            i=i+1

            se.add(f_name[0:4])

            #if (int(f_name[0:4])) not in sed:

            sed[(f_name[0:4])]=0

                #print(sed[int(f_name[0:4])])

            shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_train/'+f_name, '/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+str((f_name[0:4]))+'_'+str(sed[(f_name[0:4])])+'.jpg')

        else:

            i=i+1

            #print(i)

            sed[(f_name[0:4])]=sed[(f_name[0:4])]+1

            shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_train/'+f_name, '/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+str((f_name[0:4]))+'_'+str(sed[(f_name[0:4])])+'.jpg')

    else:

        print(int(f_name[0:4]))

            #shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_train/'+f_name, '/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+str((f_name[0:4]))+'_'+str(sed[(f_name[0:4])])+'.jpg')

print(i)

84
1451
129
129
1451
84
12930

import os

​

import shutil

​

se1=set()

​

sed1={}

​

for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1'):

​

    if f_name[0:4] not in se1:

​

        se1.add(f_name[0:4])

        sed1[f_name[0:4]]=1

​

        #print(int(f_name[0:4]))

​

    else:

​

        #print(f_name)

​

        

​

        sed1[f_name[0:4]]=sed1[f_name[0:4]]+1

​

#print(sed1)

sed2={}

​

i=0

print(len(sed1))

s=0

for key,value in sed1.items():

    s=s+value

    

print('s',s)

for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1'):

​

    i=i+1

​

    so=int(f_name[5:-4])

​

    s1=sed1[f_name[0:4]]

    #print(f_name[0:4],s1)

    #print("i",i)

​

    #print(s1)

​

    #sed2[f_name]=str(f_name[0:5])+str((so+1)%sed1[f_name[0:4]])+'.jpg'

​

    shutil.copy('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+f_name, '/home/ws2/Downloads/Market1501_train_test/nonocc_train_diffpose_v1/'+str(f_name[0:5])+str((so+1)%sed1[f_name[0:4]])+'.jpg')

    #shutil.copy('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+f_name, '/home/ws2/Downloads/Market1501_train_test/ave/'+f_name[:-4]+'.jpg')

print(s1)

​

#print(len(se1))

​

#print(len(sed1))

748
s 12930
10

p=list(se1)

aa={}

i=0

for x in p:

    aa[x]=i

    i=i+1

    

print(len(p))

748


# In[33]:


import numpy
a=[]
for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1'):
    a.append(f_name[:-4])
numpy.save('/home/ws2/Downloads/Market1501_train_test/market_v1_iden', a)
    


# In[34]:


a=numpy.load('/home/ws2/Downloads/Market1501_train_test/market_v1_iden.npy')
print(a)


# In[ ]:




