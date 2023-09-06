
import pandas as pd
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import seaborn as sns
from PIL import Image
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import autokeras as ak
from sklearn.utils import resample

# Dump all images into a folder and specify the path:
data_dir = os.getcwd() + "D:\Practicals\CGC\all_images"

# Path to destination directory where we want subfolders
dest_dir = os.getcwd() + "D:\Practicals\CGC\reorganized"

# Read the csv file containing image names and corresponding labels
skin_df2 = pd.read_csv("D:\Practicals\CGC\HAM10000_metadata.csv")
print(skin_df2['dx'].value_counts())


label=skin_df2['dx'].unique().tolist()  #Extract labels into a list
label_images = []
s_1=[]
# Copy images to new folders
for i in label:
    # print(str(i))
    # os.mkdir( "D:\Practicals\CGC\\reorganized"+"\\"+str(i))
    sample = skin_df2[skin_df2['dx'] == i]['image_id']
    label_images.extend(sample)

    # for id in label_images:
    #     shutil.copyfile(("D:\Practicals\CGC\\all_images\\" + "\\"+ id +".jpg"), ( "D:\Practicals\CGC\\reorganized\\" + i + "\\"+id+".jpg"))
    label_images=[] #empty list



#Define datagen. Here we can define any transformations we want to apply to images
datagen = ImageDataGenerator()


train_data_keras = datagen.flow_from_directory(directory="D:\Practicals\CGC\\reorganized",
                                         class_mode='categorical',
                                         batch_size=16,  #16 images at a time
                                         target_size=(32,32))  #Resize images

#We can check images for a single batch.
x, y = next(train_data_keras)

# for i in range (0,15):
#     image = x[i].astype(int)
#     plt.imshow(image)
#     plt.show()


np.random.seed(42)

SIZE=32

le = LabelEncoder()
le.fit(skin_df2['dx'])
LabelEncoder()
print(list(le.classes_))



skin_df2['label'] = le.transform(skin_df2["dx"]) 
print(skin_df2.sample(10))


# Data distribution visualization
fig = plt.figure(figsize=(15,10))



ax1 = fig.add_subplot(221)
skin_df2['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type')

ax2 = fig.add_subplot(222)
skin_df2['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex')


ax3 = fig.add_subplot(223)
skin_df2['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count',size=12)
ax3.set_title('Localization')


ax4 = fig.add_subplot(224)
sample_age = skin_df2[pd.notnull(skin_df2['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='red');
ax4.set_title('Age')

plt.tight_layout()
plt.show()

# Distribution of data into various classes 

print(skin_df2['label'].value_counts())

#Balance data.
# Many ways to balance data... you can also try assigning weights during model.fit
#Separate each classes, resample, and combine back into single dataframe

df_0 = skin_df2[skin_df2['label'] == 0]
df_1 = skin_df2[skin_df2['label'] == 1]
df_2 = skin_df2[skin_df2['label'] == 2]
df_3 = skin_df2[skin_df2['label'] == 3]
df_4 = skin_df2[skin_df2['label'] == 4]
df_5 = skin_df2[skin_df2['label'] == 5]
df_6 = skin_df2[skin_df2['label'] == 6]


n_samples=500 
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced, 
                              df_2_balanced, df_3_balanced, 
                              df_4_balanced, df_5_balanced, df_6_balanced])



print(skin_df_balanced['label'].value_counts())


# image_path = {os.path.splitext(os.path.basename(x))[0]: x
#                      for x in glob(os.path.join('D:\Practicals\CGC\reorganized\\', '*', '*.jpg'))}
# print(image_path)
print(label)


# for i in range(len(label)):
#         for l in label:
#             # dir_path=f"D:\Practicals\CGC\\reorganized\{l}\\"
#             # res = []
#             # for path in os.listdir(dir_path):
#             # # check if current path is a file
#             #     if os.path.isfile(os.path.join(dir_path, path)):
#             #         res.append(path)

#             dir_path = f"D:\Practicals\CGC\\reorganized\{l}\\*.*"
#             res = glob.glob(dir_path)
            
#             image_path.append(res)
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('D:\Practicals\CGC', '*', '*.jpg'))}



skin_df_balanced['path'] = skin_df2['image_id'].map(image_path.get)

print(skin_df_balanced)

skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))


# n_samples = 5  # number of samples for plotting
# # Plotting
# fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
# for n_axs, (type_name, type_rows) in zip(m_axs, 
#                                          skin_df_balanced.sort_values(['dx']).groupby('dx')):
#     n_axs[0].set_title(type_name)
#     for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
#         c_ax.imshow(c_row['image'])
#         c_ax.axis('off')

#Convert dataframe column of images into numpy array
X = np.asarray(skin_df_balanced['image'].tolist())
X = X/255. # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y=skin_df_balanced['label'] #Assign label values to Y
Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem
#Split to training and testing. Get a very small dataset for training as we will be 
# fitting it to many potential models. 
x_train_auto, x_test_auto, y_train_auto, y_test_auto = train_test_split(X, Y_cat, test_size=0.2, random_state=42)

#Further split data into smaller size to get a small test dataset. 
x_unused, x_valid, y_unused, y_valid = train_test_split(x_test_auto, y_test_auto, test_size=0.2, random_state=42)

#Define classifier for autokeras. Here we check 25 different models, each model 25 epochs
clf = ak.ImageClassifier(max_trials=25) #MaxTrials - max. number of keras models to try
clf.fit(x_train_auto, y_train_auto, epochs=50)


#Evaluate the classifier on test data
_, acc = clf.evaluate(x_valid, y_valid)
print("Accuracy = ", (acc * 100.0), "%")

# get the final best performing model
model = clf.export_model()
print(model.summary())

#Save the model
model.save('cifar_model.h5')

score = model.evaluate(x_valid, y_valid)
print('Test accuracy:', score[1])