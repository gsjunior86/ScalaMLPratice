package br.org.gsj.ml.spark.linear_regression

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.collection.mutable.ListBuffer

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

    val independent_variables = Array("season", "yr", "mnth", "holiday",
      "weekday", "workingday", "weathersit", "temp", "atemp",
      "hum", "windspeed")
      
    val dependent_variable = Array("cnt")
    
    val bike_sharing_df = bsDay_df.select(independent_variables.union(dependent_variable).map(col): _*)
    
    val listRes = new ListBuffer[String]

    for (i <- bike_sharing_df.columns) {
      listRes+=("Correlation to CNT for: " + i + ", " + bsDay_df.stat.corr("cnt", i))
    }
    
    listRes.foreach(println)

//    bsDay_df.printSchema()

  }

}