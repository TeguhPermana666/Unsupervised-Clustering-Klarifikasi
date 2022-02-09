import pandas as pd
from sklearn import cluster
#membaca sebuah dataset menjadi data frame
df=pd.read_csv("unsupervised\Mall_Customers.csv")
#tampilkan 3 baris pertama
print(df.head(3))
#preprocesing dataframe
"""
1.Rename nama colums supaya lebih seragam
2.ubah data kategorik ke dalam sebuah data numerik
"""
df=df.rename(columns={"Gender":"gender","Age":"age",
                      "Annual Income (k$)":"annual_income",
                      "Spending Score (1-100)":"spending_score"})
#Ubah data kategorik menjadi numerik
# df=pd.get_dummies(df)=>hotdecoding
df["gender"].replace(["Female","Male"],[0,1],inplace=True)
#artinya sebuah data frame pada geder yang berisikan data female dan male akan di convert ke numeric 0,1 
#dimana convert tersebut terjadi dengan inplace=true yg berarti data frame tersebut akan berubah
print(df.head(5))

#menghilangkan kolom gender dan id karena mengganggu proses clustering
X=df.drop(["CustomerID","gender"],axis=1)
print(X.head())

#membuat list yang berisikan inertia
from sklearn.cluster import KMeans
clusters=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km=km.fit(X)
    #dihitung inertianya di setiap klaster=>seberapa jarak anatara komponen(sampel) tiap klaster (k)
    clusters.append(km.inertia_)#memasukan setiap nilai inertia k kluster pada variabel clusters
print(clusters)
#visualisasi untuk mengerahui titik k pada teknik elbow
import matplotlib.pyplot as plt
import seaborn as sns
fig,ax=plt.subplots(figsize=(10,8))
sns.lineplot(x=list(range(1,11)), y=clusters)
ax.set_title("Cari elbow")
ax.set_xlabel("Clusters")
ax.set_ylabel("Inertia")
plt.show()
#didapatlah sebuah nilai k yakni 5, karena di 5 dia tidak turun secara signifikan
"""
Maka model kmeans skrng di iterasi sebanyak 5 kali untuk mendapatkan model yg naik dalam melatih model
"""
#membuat object kmeans
km2=KMeans(n_clusters=5)#karena pada iterasi ke 5 menghasilkan klaster variance yang bagus, maka iterasi akan dilakukan sebanyak 5 kali
km2=km2.fit(X)
#menambahka kolom label pada dataset
X["Labels"]=km2.labels_
#membuat plot Kmeans dengan 5 klaster
plt.figure(figsize=(8,4))
sns.scatterplot(X["annual_income"],X["spending_score"],hue=X["Labels"],
                palette=sns.color_palette("hls",5))
plt.title("KMeans dengan 5 clusters")
plt.show()
print(X)

#sbnrnya bisa di train_test split, biar cpt aja langsung
y=X.drop(['Labels'],axis=1)
model=km2.fit(X,y)
predicts=model.predict(X)
print(predicts)

#cetak semua label
print('Label->0\n')
print(X[X['Labels']==0])


print('Label->1\n')
print(X[X['Labels']==1])

print('Label->2\n')
print(X[X['Labels']==2])

print('Label->3\n')
print(X[X['Labels']==3])

print('Label->4\n')
print(X[X['Labels']==4])
# X.to_csv("D:/Machine Learning/Coding/unsupervised/Hasil_Clusters.csv")
# print(km2)
