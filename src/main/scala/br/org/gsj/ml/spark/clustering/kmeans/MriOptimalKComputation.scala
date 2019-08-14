package br.org.gsj.ml.spark.clustering.kmeans

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
import vegas._
import java.io.PrintWriter
import java.io.File

/**
 * plot the optimal number of clusters K based on the K-means cost function for a range of K
 */
object MriOptimalKComputation {
    case class Image(data: Byte)

  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder().appName("mriClass").master("local[*]").getOrCreate()
    val mri_healthy_brain_image = "src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain.png"
    
    val image_df = spark.read.format("image").load(mri_healthy_brain_image).select(col("image.*"))
    image_df.show
    image_df.printSchema
   
    
    val data = image_df.rdd.collect().map(f => f(5))
    
    val data_array: Array[Byte] = data(0).asInstanceOf[Array[Byte]]
     import spark.implicits._
    
    val transposed_df = spark.sparkContext.parallelize(data_array).map(f => Image(f)).toDF

    transposed_df.show
    
    val features_col = Array("data")
    val vector_assembler = new VectorAssembler()
    .setInputCols(features_col)
    .setOutputCol("features")
    
    val mri_healthy_brain_df = vector_assembler.transform(transposed_df).select("features")
    
    
    val cost = new Array[Double](21)
    
    for(k<-2 to 20){
      val kmeans = new KMeans().setK(k).setSeed(1L).setFeaturesCol("features")
      val model =  kmeans.fit(mri_healthy_brain_df.sample(false,0.1))
      cost(k) = model.computeCost(mri_healthy_brain_df)
    }
    
    var plotSeq = 
        cost.zipWithIndex.map{case (l,i)=> collection.mutable.Map("clusters" -> i, "cost" -> l)}
    .map(p => p.retain((k,v) => v != 0)).map(f => Map(f.toSeq:_*))
        
    val writer = new PrintWriter(new File("src/main/resources/plot/kmeans/plot.html"))

    val plot = Vegas("Kmeans Cost").withData(plotSeq).
      encodeX("clusters", Nom).
      encodeY("cost", Quant).
      mark(Line)

     writer.write(plot.html.pageHTML("plot"))
     writer.close()
 
  
    
  }
  
}