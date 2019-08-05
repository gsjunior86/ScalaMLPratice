package br.org.gsj.ml.spark.clustering

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
import vegas._
import java.io.PrintWriter
import java.io.File

object MriClustering {
  
  case class Image(data: Byte)
  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder().appName("mriClass").master("local[*]").getOrCreate()
    val mri_healthy_brain_image = "src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain.png"
    
    val image_df = spark.read.format("image").load(mri_healthy_brain_image).select(col("image.*"))
    image_df.show
    image_df.printSchema
    import spark.implicits._
    
    val data = image_df.rdd.collect().map(f => f(5))
    
    val data_array: Array[Byte] = data(0).asInstanceOf[Array[Byte]]
    
    val transposed_df = spark.sparkContext.parallelize(data_array).map(f => Image(f)).toDF

    transposed_df.show
    
    val features_col = Array("data")
    val vector_assembler = new VectorAssembler()
    .setInputCols(features_col)
    .setOutputCol("features")
    
    val mri_healthy_brain_df = vector_assembler.transform(transposed_df).select("features")
    
    val k = 5
    val kmeans = new KMeans().setK(k).setSeed(12345).setFeaturesCol("features")
    val kmeans_model = kmeans.fit(mri_healthy_brain_df)   
    val kmeans_centers = kmeans_model.clusterCenters
    println("Cluster Centers --------")
    for(k <- kmeans_centers)
      println(k)

  }
  
}