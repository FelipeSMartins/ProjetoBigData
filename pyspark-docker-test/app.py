from pyspark.sql import SparkSession

# Cria SparkSession
spark = SparkSession.builder.appName("TestePySpark").getOrCreate()

# Teste simples
df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "nome"])
df.show()

spark.stop()
