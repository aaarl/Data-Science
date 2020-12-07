#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[25]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# for pretty printing
def printDf(sprkDF): 
    newdf = sprkDF.toPandas()
    from IPython.display import display, HTML
    return HTML(newdf.to_html())

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Spark libs
from pyspark.sql.session import SparkSession

# helper functions '(MH)' Need to move helpers folder from exercise to DS-Project
from helpers.helper_functions import translate_to_file_string

from pyspark.sql.types import BooleanType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from socket import gethostname, gethostbyname
from helpers.helper_functions import translate_to_file_string

from pyspark.sql import DataFrameReader
from pyspark.sql import SparkSession
from pyspark.ml.feature import IndexToString, Normalizer, StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from helpers.helper_functions import translate_to_file_string

# for pretty printing
def printDf(sprkDF): 
    newdf = sprkDF.toPandas()
    from IPython.display import display, HTML
    return HTML(newdf.to_html())


# ## Data Input

# In[26]:


inputFile = translate_to_file_string("../DS-Project/squark_offer_training_data.csv")


# ## Spark session creation

# In[27]:


spark = (SparkSession
       .builder 
       .master("local") 
       .appName("PredictOffers")
       .getOrCreate())


# ## Read data

# In[31]:


df = spark.read.option("header", "true")        .option("inferSchema", "true")        .option("delimiter", ",")        .csv(inputFile) 
 #     .withColumn("Contract", expr("CAT").cast(String()))
print(df.printSchema())


# In[32]:


## Prepare training and test data


# In[33]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in list(set(df.columns)-set(['LEFTOVER'])) ]


pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)

featureCols = df_r.columns.copy() 
#featureCols.remove("LEAVE")
#featureCols.remove("COLLEGE")
#featureCols.remove("REPORTED_SATISFACTION")
#featureCols.remove("REPORTED_USAGE_LEVEL")
#featureCols.remove("CONSIDERING_CHANGE_OF_PLAN")
#featureCols.remove("OVERAGE")
#featureCols.remove("LEFTOVER")
#featureCols.remove("HOUSE")
#featureCols.remove("HANDSET_PRICE")
#featureCols.remove("OVER_15MINS_CALLS_PER_MONTH")
#featureCols.remove("AVERAGE_CALL_DURATION")
print(featureCols)

#assember builds col vector for data set
assembler =  VectorAssembler(outputCol="features", inputCols=list(featureCols))

labelIndexer = StringIndexer().setInputCol("Contract").setOutputCol("CAT").fit(df)
labeledData = labelIndexer.transform(df_r)
#TODO transform the data with the other indexer 
labeledPointData = assembler.transform(labeledData)
labeledPointData.show()

df_r.show()
print(df_r.printSchema())


# In[ ]:




