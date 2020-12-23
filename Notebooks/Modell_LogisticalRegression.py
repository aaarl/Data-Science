#!/usr/bin/env python
# coding: utf-8

# # Logistical Regression

# In[1]:

from pyspark.sql.types import BooleanType
from pyspark.ml.feature import IndexToString, Normalizer, StringIndexer, VectorAssembler, VectorIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics, RegressionMetrics
from pyspark.mllib.util import MLUtils
import pandas as pd
from IPython.display import display, HTML

#Change Column Names
def delete_space(df):
    names = df.schema.names
    for name in names:
        newName = name.replace(" ","")
        df = df.withColumnRenamed(name, newName)
    return df


# In[2]:


inputFile = "hdfs:///data/data.csv"


# Spark session creation 

# In[3]:


spark = (SparkSession
       .builder
       .master("yarn")
       .appName("Modell_LogisticalRegression")
       .getOrCreate())


# DataFrame creation using an ifered Schema 

# In[4]:


# create a DataFrame using an ifered Schema 
df = spark.read.option("header", "true")        .option("inferSchema", "true")        .option("delimiter", ";")        .csv(inputFile) 

df = delete_space(df)
df_orig = df
df = df.where("MonthlyCharges Between 22 AND 95")
df = df.where("TotalCharges IS NOT NULL")


# ## Prepare training and test data.
# 
# ### Use this for Pandas Dataframe

# In[5]:



#Create Pandas DataFrame
df_pandas = df.toPandas()
df_pandas_cat = df.toPandas()
#Pandas Indexing Method to Integer or Category Datatype
pandasCol = list(df_pandas)
for col in pandasCol:
    if df_pandas[col].dtypes=='object':
        if not col == "Contract":
            #Categorize
            df_pandas_cat[col]= pd.Categorical(pd.factorize(df_pandas_cat[col])[0])
            #ToInteger
            df_pandas[col]= pd.factorize(df_pandas[col])[0]

#Define whicht Columns should be normalized
newCols = []
for col in pandasCol:
    if not col == "Contract":
        newCols.append(col)

#Normalize the selected Columns
df_pandas[newCols]=(df_pandas[newCols]-df_pandas[newCols].min())/(df_pandas[newCols].max()-df_pandas[newCols].min())



#Write Pandas Dataframe Back to Spark Dataframe
df_temp = spark.createDataFrame(df_pandas)
df = df_temp
df.printSchema()
# Create Indexer for Contract (Still needed)
contractIndexer = StringIndexer().setInputCol("Contract").setOutputCol("Contract_Int").fit(df)

#Create FeatureCols 
featureCols = df.columns.copy()
featureCols.remove("Contract")
featureCols.remove("CustomerID")


# ### Create Indexer

# In[6]:


#Comment the Following if Pandas-Dataset is used
# IDIndexer = StringIndexer().setInputCol("CustomerID").setOutputCol("CustomerID_Int").fit(df)
# genderIndexer = StringIndexer().setInputCol("Gender").setOutputCol("Gender_Int").fit(df)
# seniorIndexer = StringIndexer().setInputCol("SeniorCitizen").setOutputCol("SeniorCitizen_Int").fit(df)
# partnerIndexer = StringIndexer().setInputCol("Partner").setOutputCol("Partner_Int").fit(df)
# DependentsIndexer = StringIndexer().setInputCol("Dependents").setOutputCol("Dependents_Int").fit(df)
# tenureIndexer = StringIndexer().setInputCol("Tenure").setOutputCol("Tenure_Int").fit(df)
# phoneIndexer = StringIndexer().setInputCol("PhoneService").setOutputCol("PhoneService_Int").fit(df)
# multipleIndexer = StringIndexer().setInputCol("MultipleLines").setOutputCol("MultipleLines_Int").fit(df)
# internetIndexer = StringIndexer().setInputCol("InternetService").setOutputCol("InternetService_Int").fit(df)
# onlineSecurityIndexer = StringIndexer().setInputCol("OnlineSecurity").setOutputCol("OnlineSecurity_Int").fit(df)
# onlineBackupIndexer = StringIndexer().setInputCol("OnlineBackup").setOutputCol("OnlineBackup_Int").fit(df)
# deviceIndexer = StringIndexer().setInputCol("DeviceProtection").setOutputCol("DeviceProtection_Int").fit(df)
# techIndexer = StringIndexer().setInputCol("TechSupport").setOutputCol("TechSupport_Int").fit(df)
# streamingTVIndexer = StringIndexer().setInputCol("StreamingTV").setOutputCol("StreamingTV_Int").fit(df)
# streamingMoviesIndexer = StringIndexer().setInputCol("StreamingMovies").setOutputCol("StreamingMovies_Int").fit(df)
# contractIndexer = StringIndexer().setInputCol("Contract").setOutputCol("Contract_Int").fit(df)
# paperlessIndexer = StringIndexer().setInputCol("PaperlessBilling").setOutputCol("PaperlessBilling_Int").fit(df)
# paymentIndexer = StringIndexer().setInputCol("PaymentMethod").setOutputCol("PaymentMethod_Int").fit(df)
# monthlyIndexer = StringIndexer().setInputCol("MonthlyCharges").setOutputCol("MonthlyCharges_Int").fit(df)
# totalIndexer = StringIndexer().setInputCol("TotalCharges").setOutputCol("TotalCharges_Int").fit(df)


# In[7]:


# #Comment this if Pandas Dataset is used
# featureCols = df.columns.copy()
# for col in featureCols:
#     if not col == "Tenure" and not col == "MonthlyCharges" and not col == "TotalCharges":
#         featureCols.remove(col)
#         colname = col +"_Int"
#         featureCols = featureCols + [colname]
#     else:
#         featureCols.remove(col)
#         featureCols = featureCols + [col]

# featureCols.remove("Contract_Int")
# featureCols.remove("CustomerID_Int")
# featureCols.remove("Gender")
# featureCols = featureCols + ["Gender_Int"]


# In[8]:


assembler =  VectorAssembler(outputCol="features", inputCols=list(featureCols))
#, handleInvalid="keep"

featureIndexer = VectorIndexer(inputCol="features",outputCol="indexedFeatures", maxCategories=6) 
#, handleInvalid="keep"

predConverter = IndexToString(inputCol="prediction",outputCol="predictedLabel",labels=contractIndexer.labels)

scaler = StandardScaler(inputCol="indexedFeatures", outputCol="scaledFeatures",withStd=True, withMean=False)


# ## Logistic Regression

# In[9]:


lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="Contract_Int")
paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [100, 120 , 150])                                 .addGrid(lr.regParam, [0.1, 0.3, 0.5])                                  .addGrid(lr.standardization, [True, False])                                  .addGrid(lr.elasticNetParam, [0, 1])                                  .build()


# ### Create Train and Test Datasets

# In[10]:


splits = df.randomSplit([0.9, 0.1 ], 12345)
train = splits[0]
test = splits[1]


# ### Build the Pipeline

# In[11]:


#Use This for Pandas-Dataframe
pipeline = Pipeline(stages= [contractIndexer, assembler, featureIndexer, scaler, lr, predConverter])


# In[12]:


##Use This for Spark Dataframe
# pipeline = Pipeline(stages= [genderIndexer, seniorIndexer, partnerIndexer, DependentsIndexer, phoneIndexer, multipleIndexer, internetIndexer, onlineSecurityIndexer, onlineBackupIndexer, deviceIndexer, techIndexer, streamingTVIndexer, streamingMoviesIndexer, contractIndexer, paperlessIndexer, paymentIndexer, assembler, featureIndexer, scaler, lr, predConverter])


# ### Build the Evaluator and Cross Validator

# In[13]:


evaluator =  RegressionEvaluator(labelCol="Contract_Int", metricName="rmse")

cv = CrossValidator(estimator=pipeline,evaluator=evaluator,estimatorParamMaps=paramGrid,numFolds=3, parallelism=2)


# ### Train the Model

# In[14]:


cvModel = cv.fit(train)


# In[15]:


#stages[19] for Spark Dataframe ;  stages[4] for Pandas Dataframe 
treeModel = cvModel.bestModel.stages[4]
print("Learned classification tree model:\n",treeModel)
print("Best Params: \n", treeModel.explainParams())


# In[16]:


predictions = cvModel.transform(test)
predictions.select("prediction", "Contract_Int", "predictedLabel", "Contract", "features").show()


# In[17]:


new_df = predictions.groupBy("prediction").count()
new_df.show()
new_train_df = train.groupBy("Contract").count()
new_train_df.show()


# In[18]:


accuracy = evaluator.evaluate(predictions)
print("Test Error = " ,(1.0 - accuracy))


# In[22]:


predictionAndLabels = predictions.select("prediction", "Contract_Int").rdd.map(lambda p: [p[0], p[1]]) # Map to RDD prediction|label
metrics =  MulticlassMetrics(predictionAndLabels)


# In[23]:


confusion = metrics.confusionMatrix()


# In[24]:


labels = predictionAndLabels.map(lambda x: x[1]).distinct().collect()
for label in  labels:
  print("Class %f precision = %f\n" % (label , metrics.precision(label)))
  print("Class %f recall = %f\n" % (label, metrics.recall(label)))
  print("Class %f F1 score = %f\n" % (label, metrics.fMeasure( label)))


# In[26]:


print("Recall = %s" % metrics.recall(1.0))
print("Precision = %s" % metrics.precision(1.0))
print("Accuracy = %s" % metrics.accuracy) 
print("F1 = %s" % metrics.fMeasure(1.0))
