# Algorithmics for Data Mining
# Deliverable 3: Detecting Intrusions Using Data Mining Techniques
# Ali Arabyarmohammadi
# June 2022




# In[1]:

# Load Libraries
import warnings

from IPython import get_ipython

warnings.filterwarnings("ignore")
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

from zoofs import ParticleSwarmOptimization
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier



# In[2]:


training_df = pd.read_csv('KDDTrain+.csv', header=None)
testing_df = pd.read_csv('KDDTest+.csv', header=None)


# In[3]:


training_df.head()


# In[4]:


testing_df.head()


# In[5]:


columns = [
   'duration',
   'protocol_type',
   'service',
   'flag',
   'src_bytes',
   'dst_bytes',
   'land',
   'wrong_fragment',
   'urgent',
   'hot',
   'num_failed_logins',
   'logged_in',
   'num_compromised',
   'root_shell',
   'su_attempted',
   'num_root',
   'num_file_creations',
   'num_shells',
   'num_access_files',
   'num_outbound_cmds',
   'is_host_login',
   'is_guest_login',
   'count',
   'srv_count',
   'serror_rate',
   'srv_serror_rate',
   'rerror_rate',
   'srv_rerror_rate',
   'same_srv_rate',
   'diff_srv_rate',
   'srv_diff_host_rate',
   'dst_host_count',
   'dst_host_srv_count',
   'dst_host_same_srv_rate',
   'dst_host_diff_srv_rate',
   'dst_host_same_src_port_rate',
   'dst_host_srv_diff_host_rate',
   'dst_host_serror_rate',
   'dst_host_srv_serror_rate',
   'dst_host_rerror_rate',
   'dst_host_srv_rerror_rate',
   'outcome',
   'difficulty'
]
training_df.columns = columns
testing_df.columns = columns


# In[6]:


print("Training set has {} rows.".format(len(training_df)))
print("Testing set has {} rows.".format(len(testing_df)))


# In[7]:


training_outcomes=training_df["outcome"].unique()
testing_outcomes=testing_df["outcome"].unique()
print("The training set has {} possible outcomes \n".format(len(training_outcomes)) )
print(", ".join(training_outcomes)+".")
print("\nThe testing set has {} possible outcomes \n".format(len(testing_outcomes)))
print(", ".join(testing_outcomes)+".")


# In[8]:


# A list ot attack names that belong to each general attack type
dos_attacks=["snmpgetattack","back","land","neptune","smurf","teardrop","pod","apache2","udpstorm","processtable","mailbomb"]
r2l_attacks=["snmpguess","worm","httptunnel","named","xlock","xsnoop","sendmail","ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]
u2r_attacks=["sqlattack","buffer_overflow","loadmodule","perl","rootkit","xterm","ps"]
probe_attacks=["ipsweep","nmap","portsweep","satan","saint","mscan"]

# Our new labels
classes=["Normal","Dos","R2L","U2R","Probe"]

#Helper function to label samples to 5 classes
def label_attack (row):
    if row["outcome"] in dos_attacks:
        return classes[1]
    if row["outcome"] in r2l_attacks:
        return classes[2]
    if row["outcome"] in u2r_attacks:
        return classes[3]
    if row["outcome"] in probe_attacks:
        return classes[4]
    return classes[0]


#We combine the datasets temporarily to do the labeling 
test_samples_length = len(testing_df)
df=pd.concat([training_df,testing_df])
df["Class"]=df.apply(label_attack,axis=1)


# The old outcome field is dropped since it was replaced with the Class field, the difficulty field will be dropped as well.
df=df.drop("outcome",axis=1)
df=df.drop("difficulty",axis=1)

# we again split the data into training and test sets.
training_df= df.iloc[:-test_samples_length, :]
testing_df= df.iloc[-test_samples_length:,:]


# In[9]:


training_outcomes=training_df["Class"].unique()
testing_outcomes=testing_df["Class"].unique()
print("The training set has {} possible outcomes \n".format(len(training_outcomes)) )
print(", ".join(training_outcomes)+".")
print("\nThe testing set has {} possible outcomes \n".format(len(testing_outcomes)))
print(", ".join(testing_outcomes)+".")


# In[10]:


# Helper function for scaling continous values
def minmax_scale_values(training_df,testing_df, col_name):
    scaler = MinMaxScaler()
    scaler = scaler.fit(training_df[col_name].values.reshape(-1, 1))
    train_values_standardized = scaler.transform(training_df[col_name].values.reshape(-1, 1))
    training_df[col_name] = train_values_standardized
    test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
    testing_df[col_name] = test_values_standardized
    
    
#Helper function for one hot encoding
def encode_text(training_df,testing_df, name):
    training_set_dummies = pd.get_dummies(training_df[name])
    testing_set_dummies = pd.get_dummies(testing_df[name])
    for x in training_set_dummies.columns:
        dummy_name = "{}_{}".format(name, x)
        training_df[dummy_name] = training_set_dummies[x]
        if x in testing_set_dummies.columns :
            testing_df[dummy_name]=testing_set_dummies[x]
        else :
            testing_df[dummy_name]=np.zeros(len(testing_df))
    training_df.drop(name, axis=1, inplace=True)
    testing_df.drop(name, axis=1, inplace=True)
    
    
sympolic_columns=["protocol_type","service","flag"]
label_column="Class"
for column in df.columns :
    if column in sympolic_columns:
        encode_text(training_df,testing_df,column)
    elif not column == label_column:
        minmax_scale_values(training_df,testing_df, column)


# In[11]:


training_df.head(5)


# In[12]:


testing_df.head(5)


# In[13]:


x,y=training_df,training_df.pop("Class").values
x=x.values
x_test,y_test=testing_df,testing_df.pop("Class").values
x_test=x_test.values


# ## SET Attack and Normal Classes

# In[14]:


# Classes[0] = 'Normal'
# Classes[1] = 'Dos'
# Classes[2] = 'R2L'
# Classes[3] = 'U2R'
# Classes[4] = 'Probe'
y0=np.ones(len(y),np.int8)
y0[np.where(y==classes[0])]=0
y0_test=np.ones(len(y_test),np.int8)
y0_test[np.where(y_test==classes[0])]=0


# In[15]:


y = y0
y_test = y0_test


# ## DecisionTreeClassifier and ParticleSwarmOptimization

# In[21]:



def numpy2dataframe(nparray):
    panda_df = pd.DataFrame(data = nparray,
                            index = ['Row_' + str(i + 1) 
                            for i in range(nparray.shape[0])],
                            columns = ['Column_' + str(i + 1) 
                            for i in range(nparray.shape[1])])
    return panda_df


def objective_function_topass(model,X_train, y_train, X_valid, y_valid):      
    model.fit(X_train,y_train)  
    P=accuracy_score(y_valid, model.predict(X_valid))
    
    return P



algo_object=ParticleSwarmOptimization(objective_function_topass,n_iteration=4, population_size=4,minimize=False)


xtrain_df = numpy2dataframe(x)
xtest_df = numpy2dataframe(x_test)

clf = DecisionTreeClassifier(random_state=0)                                      
best_feature_list = algo_object.fit(clf, xtrain_df, pd.DataFrame(y), xtest_df, pd.DataFrame(y_test), verbose=True)
algo_object.plot_history()


# In[22]:


print("Best features:\n")
best_feature_list




