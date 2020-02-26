#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
 # Get .txt files
for f_name in os.listdir('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_occ'):
    p=(int(f_name[:-8])+1)%125000
    shutil.copy('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_occ/'+f_name, '/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_occ_wrong/'+str(p)+'.png')
    

for f_name in os.listdir('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_occ'):
    if int(f_name[:-8])%4==0 and int(f_name[:-8])!=0:
        #print(f_name)
        shutil.copy('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_occ/'+f_name, '/home/ws2/Downloads/occluded_person_reidentification/AiC_test_train/crops_occ/test')
    else:
        shutil.copy('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_occ/'+f_name, '/home/ws2/Downloads/occluded_person_reidentification/AiC_test_train/crops_occ/train')
        
        
        
for f_name in os.listdir('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_occ_wrong'):
    if int(f_name[:-4])%4==0 and int(f_name[:-4])!=0:
        #print(f_name)
        shutil.copy('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_occ_wrong/'+f_name, '/home/ws2/Downloads/occluded_person_reidentification/AiC_test_train/crops_occ_wrong/test')
    else:
        shutil.copy('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_occ_wrong/'+f_name, '/home/ws2/Downloads/occluded_person_reidentification/AiC_test_train/crops_occ_wrong/train')
        
        
for f_name in os.listdir('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_nonocc'):
    if int(f_name[:-4])%4==0 and int(f_name[:-4])!=0:
        #print(f_name)
        shutil.copy('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_nonocc/'+f_name, '/home/ws2/Downloads/occluded_person_reidentification/AiC_test_train/crops_nonocc/test')
    else:
        shutil.copy('/home/ws2/Downloads/occluded_person_reidentification/AiC/crops_nonocc/'+f_name, '/home/ws2/Downloads/occluded_person_reidentification/AiC_test_train/crops_nonocc/train')


# In[26]:


import os
import shutil
a=-1
c=0
i=0
se=set()
sed={}
for f_name in os.listdir('/home/ws2/Downloads/Market-1501/bounding_box_test'):
    #print(int(f_name[0:4]))
    '''if (int(f_name[0:2]) != -1):
        print(int(f_name[0:4]))
    else:
        print(int(f_name[0:2]))'''
    i=i+1
    if (int(f_name[0:2]) != -1):
        if (int(f_name[0:4]) not in se ):
            #print(f_name[0:4])
            print(i)
            a=int(f_name[0:4])
            c=1
            se.add(int(f_name[0:4]))
            sed[int(f_name[0:4])]=1
        elif (int(f_name[0:4]) in se and sed[int(f_name[0:4])]==1):
            sed[int(f_name[0:4])]=2
            #print(int(f_name[0:4]))
            #shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_test/'+f_name, '/home/ws2/Downloads/Market1501_train_test/crops_nonocc/test/'+str(int(f_name[0:4]))+'.jpg')
            
    


# In[42]:


i=0
for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/train_occ'):
    
    if((i)%2 == 0):
        f_name1=f_name
        #print(f_name)
        #print(i)
        print('i%2=0',i)
    elif((i)%2 == 1):
        f_name2=f_name
        shutil.copy('/home/ws2/Downloads/Market1501_train_test/train_occ/'+f_name1, '/home/ws2/Downloads/Market1501_train_test/train_occ_wrong/'+str(int(f_name2[:-8]))+'.jpg')
        shutil.copy('/home/ws2/Downloads/Market1501_train_test/train_occ/'+f_name2, '/home/ws2/Downloads/Market1501_train_test/train_occ_wrong/'+str(int(f_name1[:-8]))+'.jpg')
        print('i%2=1',i)
    i=i+1
    print(i)


# In[43]:


import numpy
a=[]
i=0
for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/crops_nonocc/train'):
    a.append(int(f_name[:-4]))
    i=i+1
a=numpy.asarray(a)
print(i)
numpy.save('/home/ws2/Downloads/Market1501_train_test/image_index',a)
    


# In[44]:


a=numpy.load('/home/ws2/Downloads/Market1501_train_test/image_index.npy')
print(a)


# In[ ]:


import os
import shutil
a=-1
c=0
i=0
se=set()
sed={}
for f_name in os.listdir('/home/ws2/Downloads/Market-1501/bounding_box_test'):
    #print(int(f_name[0:4]))
    '''if (int(f_name[0:2]) != -1):
        print(int(f_name[0:4]))
    else:
        print(int(f_name[0:2]))'''
    i=i+1
    if (int(f_name[0:2]) != -1):
        if (int(f_name[0:4]) not in se ):
            #print(f_name[0:4])
            print(i)
            a=int(f_name[0:4])
            c=1
            se.add(int(f_name[0:4]))
            sed[int(f_name[0:4])]=1
        elif (int(f_name[0:4]) in se and sed[int(f_name[0:4])]==1):
            sed[int(f_name[0:4])]=2
            #print(int(f_name[0:4]))
            #shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_test/'+f_name, '/home/ws2/Downloads/Market1501_train_test/crops_nonocc/test/'+str(int(f_name[0:4]))+'.jpg')
            


# In[18]:


import os
import shutil

c=0
i=0
se=set()
sed={}
for f_name in os.listdir('/home/ws2/Downloads/Market-1501/bounding_box_train'):
    if f_name[0:4] not in se:
        se.add(f_name[0:4])
        #print(int(f_name[0:4]))
    else:
        #print(f_name)
        if (int(f_name[0:4])) not in sed:
            sed[int(f_name[0:4])]=1
        else:
            sed[int(f_name[0:4])]=sed[int(f_name[0:4])]+1
#print(sed)
min =1000
min_key=0
#print(sed)
for key,value in sed.items():
    if value==1 :
        #print(value)
        print(key)
        #min=value
        #min_key=key
        
#print(min)
#print(min_key)


# In[57]:


a=[129,1451,84]

se=set()
sed={}
for f_name in os.listdir('/home/ws2/Downloads/Market-1501/bounding_box_train'):
    if (int(f_name[0:4])) not in a:
        
        print(int(f_name[0:4]))
        if f_name[0:4] not in se:
            se.add(f_name[0:4])
            #if (int(f_name[0:4])) not in sed:
            sed[(f_name[0:4])]=0
                #print(sed[int(f_name[0:4])])
            shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_train/'+f_name, '/home/ws2/Downloads/Market1501_train_test/crops_nonocc_train1/'+str((f_name[0:4]))+'_'+str(sed[(f_name[0:4])])+'.jpg')
        else:
            sed[(f_name[0:4])]=sed[(f_name[0:4])]+1
            shutil.copy('/home/ws2/Downloads/Market-1501/bounding_box_train/'+f_name, '/home/ws2/Downloads/Market1501_train_test/crops_nonocc_train1/'+str((f_name[0:4]))+'_'+str(sed[(f_name[0:4])])+'.jpg')
        #print(int(f_name[0:4]))
        


# In[63]:


import os
import shutil
se1=set()
sed1={}
for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1'):
    if f_name[0:4] not in se1:
        se1.add(f_name[0:4])
        #print(int(f_name[0:4]))
    else:
        #print(f_name)
        if (f_name[0:4]) not in sed1:
            sed1[f_name[0:4]]=1
        else:
            sed1[f_name[0:4]]=sed1[f_name[0:4]]+1

sed2={}
i=0
for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1'):
    i=i+1
    so=int(f_name[5:-4])
    s1=sed1[f_name[0:4]]
    print(s1)
    sed2[f_name]=str(f_name[0:5])+str((so+1)%s1)+'.jpg'
    shutil.copy('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+f_name, '/home/ws2/Downloads/Market1501_train_test/nonocc_train_diffpose_v1/'+str(f_name[0:5])+str((so+1)%sed1[f_name[0:4]])+'.jpg')
print(i)
#print(len(se1))
#print(len(sed1))


# In[ ]:




