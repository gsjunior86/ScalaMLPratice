package br.org.gsj.ml.spark.linear_regression

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object BikeSharing_Multivariate {
  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder()
      .appName("bikeSharing-spark")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    val bsDay_df = spark.read.format("csv").option("header", "true")
      .load("src/main/resources/datasets/bike_sharing/day.csv")
      .withColumn("season", col("season").cast(IntegerType))
      .withColumn("yr", col("yr").cast(IntegerType))
      .withColumn("mnth", col("mnth").cast(IntegerType))
      .withColumn("holiday", col("holiday").cast(IntegerType))
      .withColumn("weekday", col("weekday").cast(IntegerType))
      .withColumn("workingday", col("workingday").cast(IntegerType))
      .withColumn("weathersit", col("weathersit").cast(IntegerType))
      .withColumn("temp", col("temp").cast(DoubleType))
      .withColumn("atemp", col("atemp").cast(DoubleType))
      .withColumn("hum", col("hum").cast(DoubleType))
      .withColumn("windspeed", col("windspeed").cast(DoubleType))
      .withColumn("casual", col("casual").cast(IntegerType))
      .withColumn("registered", col("registered").cast(IntegerType))
      .withColumn("cnt", col("cnt").cast(IntegerType))

      

       bsDay_df.printSchema()
    
  }
  
}