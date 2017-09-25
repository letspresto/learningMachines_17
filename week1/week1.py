
# coding: utf-8

# In[1]:

def encode(input_string):
    count = 1
    prev = ''
    lst = []
    for character in input_string:
        if character != prev:
            if prev:
                entry = (prev,count)
                lst.append(entry)
                #print 1st
            count = 1
            prev = character
        else:
            count += 1
    else:
        entry = (character,count)
        lst.append(entry)
    return lst


# In[2]:

def decode(lst):
    q = ""
    for character, count in lst:
        q += character * count
    return q


# In[3]:

#method call
encode("ooooooiiiissssdddkdkkdkkkdklllldll")


# In[ ]:

decode([('a', 5), ('h', 6), ('m', 7), ('u', 1), ('i', 7), ('a', 6)])


# In[ ]:



