package br.org.gsj.ml.spark.linear_regression

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import vegas._
import java.io.PrintWriter
import java.io.File
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types._

object BikeSharing {

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
    
    final_df.show
    

  }

}