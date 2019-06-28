package br.org.gsj.ml.spark.linear_regression

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

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
      listRes += ("Correlation to CNT for: " + i + ", " + bsDay_df.stat.corr("cnt", i))
    }

    listRes.foreach(println)

    val multivariate_feature_columns = Array("season", "yr", "mnth", "temp", "atemp")
    val multivariate_label_column = Array("cnt")

    val vector_assembler = new VectorAssembler()
      .setInputCols(multivariate_feature_columns)
      .setOutputCol("features")

    val bike_sharing_features_df = vector_assembler
      .transform(bike_sharing_df)
      .select("features", multivariate_label_column(0))

    bike_sharing_features_df.show

    val train_test_array_ds = bike_sharing_features_df.randomSplit(Array(0.75, 0.25), 12345)

    val linear_regression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol(multivariate_label_column(0))

    val linear_regression_model = linear_regression.fit(train_test_array_ds(0))

    println("Model Coefficients: " + linear_regression_model.coefficients)
    println("Intercept: " + linear_regression_model.intercept)
    val training_summary = linear_regression_model.summary
    println("RMSE:" + training_summary.rootMeanSquaredError)
    println("R-SQUARED: " + training_summary.r2)
    

  }

}