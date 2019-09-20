package br.org.gsj.ml.spark.linear_regression

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import vegas._
import java.io.PrintWriter
import java.io.File
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

object BikeSharing_Univariate {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("bikeSharing-spark")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    val writer = new PrintWriter(new File("src/main/resources/plot/bike_sharing/plot.html"))

    val bsDay_df = spark.read.format("csv").option("header", "true")
      .load("src/main/resources/datasets/bike_sharing/day.csv")
      .withColumn("atemp", col("atemp").cast(DoubleType))
      .withColumn("cnt", col("cnt").cast(DoubleType))

    bsDay_df.show
    bsDay_df.printSchema()

    import vegas.sparkExt._

    val plot = Vegas("Bike Sharing")
      .withDataFrame(bsDay_df).
      encodeX("atemp", Nom).
      encodeY("cnt", Quant).
      mark(Point)

    writer.write(plot.html.pageHTML("plot"))
    writer.close()

    val inputCols = Array("atemp")
    val univariate_label_column = "cnt"

    val vector_assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("features")

    val final_df = vector_assembler.transform(bsDay_df)
      .select("features", univariate_label_column)

    val train_test_array_ds = final_df.randomSplit(Array(0.75, 0.25), 12345)

    val total_count = bsDay_df.count
    val train_count = train_test_array_ds(0).count
    val test_count = train_test_array_ds(1).count

    val linear_regression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol(univariate_label_column)

    val linear_regression_model = linear_regression.fit(train_test_array_ds(0))

    println("Model Coefficients: " + linear_regression_model.coefficients)
    println("Intercept: " + linear_regression_model.intercept)
    val training_summary = linear_regression_model.summary
    println("RMSE:" + training_summary.rootMeanSquaredError)
    println("R-SQUARED: " + training_summary.r2)

    println("TRAINING DATASET DESCRIPTIVE SUMMARY: ")
    train_test_array_ds(0).describe().show()
    print("TRAINING DATASET RESIDUALS: ")
    training_summary.residuals.show

    println("Total Data: " + total_count)
    println("Train Data Count: " + train_count)
    println("Test Data Count: " + test_count)

    val test_linear_regression_predictions_df = linear_regression_model.transform(train_test_array_ds(1))
    test_linear_regression_predictions_df
      .select("prediction", univariate_label_column, "features")
      .show(10)
      
    val linear_regression_evaluator_rmse = new RegressionEvaluator()
    .setPredictionCol("prediction")
    .setLabelCol(univariate_label_column)
    .setMetricName("rmse")
    
    val linear_regression_evaluator_r2 = new RegressionEvaluator()
    .setPredictionCol("prediction")
    .setLabelCol(univariate_label_column)
    .setMetricName("r2")
    
    val rmse = linear_regression_evaluator_rmse.evaluate(test_linear_regression_predictions_df)
    val r2 = linear_regression_evaluator_r2.evaluate(test_linear_regression_predictions_df)
    
    println("RMSE on Test Data: " + rmse)
    println("R2 on Test Data: " + r2)
    
    spark.stop


  }

}