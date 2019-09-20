package br.org.gsj.ml.spark.linear_regression

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

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
      "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed")
      .map(f => col(f))

    val dependent_variable = Array("cnt").map(f => col(f))

    val bike_sharing_df = bsDay_df.select(independent_variables ++ dependent_variable: _*)
    bike_sharing_df.show()

    for (i <- bike_sharing_df.columns) {
      println("Correlation to CNT for: " + i + ": " + bike_sharing_df.stat.corr("cnt", i))
    }

    val multivariate_feature_columns = Array("season", "yr", "mnth", "temp", "atemp")
    val multivariate_label_column = "cnt"

    val vector_assembler = new VectorAssembler()
      .setInputCols(multivariate_feature_columns)
      .setOutputCol("features")

    val bike_sharing_df_features = vector_assembler
      .transform(bike_sharing_df)
      .select(col("features"), col(multivariate_label_column))

    bike_sharing_df_features.show()

    val prediction_df = bike_sharing_df_features.randomSplit(Array(0.75, 0.25))
    println(prediction_df(0).count)
    println(prediction_df(1).count)

    val linear_regression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol(multivariate_label_column)

    val linear_regression_model = linear_regression.fit(prediction_df(0))

    println("Model Coefficients: " + linear_regression_model.coefficients)
    println("Intercept: " + linear_regression_model.intercept)
    val training_summary = linear_regression_model.summary
    println("RMSE:" + training_summary.rootMeanSquaredError)
    println("R-SQUARED: " + training_summary.r2)
    
    println("TRAINING DATASET DESCRIPTIVE SUMMARY: ")
    prediction_df(0).describe().show()
    print("TRAINING DATASET RESIDUALS: ")
    training_summary.residuals.show
    
    val test_linear_regression_predictions_df = linear_regression_model.transform(prediction_df(1))
    test_linear_regression_predictions_df
      .select("prediction", multivariate_label_column, "features")
      .show(10)
      
      val linear_regression_evaluator_rmse = new RegressionEvaluator()
    .setPredictionCol("prediction")
    .setLabelCol(multivariate_label_column)
    .setMetricName("rmse")
    
    val linear_regression_evaluator_r2 = new RegressionEvaluator()
    .setPredictionCol("prediction")
    .setLabelCol(multivariate_label_column)
    .setMetricName("r2")
    
    val rmse = linear_regression_evaluator_rmse.evaluate(test_linear_regression_predictions_df)
    val r2 = linear_regression_evaluator_r2.evaluate(test_linear_regression_predictions_df)
    
    println("RMSE on Test Data: " + rmse)
    println("R2 on Test Data: " + r2)

  }

}