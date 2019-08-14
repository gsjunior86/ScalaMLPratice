package br.org.gsj.ml.spark.clustering.kmeans

import java.io.ByteArrayInputStream
import java.io.File

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.awt.image.MemoryImageSource

import javax.imageio.ImageIO
import java.awt.Toolkit
import java.awt.image.BufferedImage
import br.org.gsj.ml.scala.util.ImageUtils

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
      
      
    val mri_healthy_brain_clusters_df = kmeans_model.transform(mri_healthy_brain_df)
    .select("features","prediction")
    
    val image_array = mri_healthy_brain_clusters_df.select("prediction").rdd.map(f => f.getAs[Int](0).toByte).collect()
    println(image_array.size)
    
    val photo1 = ImageIO.read(new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain.png"))
    val photo2 = ImageUtils.generateImage(photo1, image_array)
    
    ImageIO.write(photo2, "jpg", new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-test.png"))

//    val bImage = ImageIO.read(new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain_cluster.jpg"));
//    val bos = new ByteArrayOutputStream();
//    ImageIO.write(bImage, "jpg", bos );
//    val datai = bos.toByteArray();
    
//    val ims = new MemoryImageSource(256,256,data_array.map(f => f.toInt),0,256)
//    val image = Toolkit.getDefaultToolkit().createImage(ims)
//    
//    val bi = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
//    bi.getGraphics().drawImage(image,0,0, null);
//    ImageIO.write(bi, "JPG", new File("src/main/resources/datasets/clustering/data/mri-images-data/mri-healthy-brain_cluster.jpg"));

  //http://otfried.org/scala/image.html

  }
  

  
}